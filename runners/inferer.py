import os
import time
import importlib
from pathlib import PurePath

import torch

import numpy as np

from monai.data import decollate_batch
from monai.transforms import (
    LoadImaged,
    AddChannel,
    SqueezeDimd,
    AsDiscrete,
    KeepLargestConnectedComponent,
    Compose,
    LabelFilter,
    MapLabelValue,
    Spacing,
    SqueezeDim
)
from monai.metrics import DiceMetric, HausdorffDistanceMetric

from data_utils.io import save_img
import matplotlib.pyplot as plt


def infer(model, data, model_inferer, device):
    model.eval()
    with torch.no_grad():
        output = model_inferer(data['image'].to(device))
        output = torch.argmax(output, dim=1)
    return output


def check_channel(inp):
    # check shape is 5
    add_ch = AddChannel()
    len_inp_shape = len(inp.shape)
    if len_inp_shape == 4:
        inp = add_ch(inp)
    if len_inp_shape == 3:
        inp = add_ch(inp)
        inp = add_ch(inp)
    return inp


def eval_label_pred(data, cls_num, device):
    # post transform
    post_label = AsDiscrete(to_onehot=cls_num)
    
    # metric
    dice_metric = DiceMetric(
        include_background=False,
        reduction="mean",
        get_not_nans=False
    )
    
    hd95_metric = HausdorffDistanceMetric(
        include_background=False,
        percentile=95,
        reduction="mean",
        get_not_nans=False
    )
    
    # batch data
    val_label, val_pred = (data["label"].to(device), data["pred"].to(device))
    
    # check shape is 5
    val_label = check_channel(val_label)
    val_pred = check_channel(val_pred)
    
    # deallocate batch data
    val_labels_convert = [
        post_label(val_label_tensor) for val_label_tensor in val_label
    ]
    val_output_convert = [
        post_label(val_pred_tensor) for val_pred_tensor in val_pred
    ]
    
    dice_metric(y_pred=val_output_convert, y=val_labels_convert)
    hd95_metric(y_pred=val_output_convert, y=val_labels_convert)

    dc_vals = dice_metric.get_buffer().detach().cpu().numpy().squeeze()
    hd95_vals = hd95_metric.get_buffer().detach().cpu().numpy().squeeze()
    return dc_vals, hd95_vals


def get_filename(data):
    return PurePath(data['image_meta_dict']['filename_or_obj']).parts[-1]


def get_label_transform(data_name, keys=['label']):
    transform = importlib.import_module(f'transforms.{data_name}_transform')
    get_lbl_transform = getattr(transform, 'get_label_transform', None)
    return get_lbl_transform(keys)


def run_infering(
        model,
        data,
        model_inferer,
        post_transform,
        args
    ):
    ret_dict = {}
    
    
    # test
    start_time = time.time()
    data['pred'] = infer(model, data, model_inferer, args.device)
    end_time  = time.time()
    ret_dict['inf_time'] = end_time-start_time
    print(f'infer time: {ret_dict["inf_time"]} sec')
    
    # post process transform
    if args.infer_post_process:
        print('use post process infer')
        applied_labels = np.unique(data['pred'].flatten())[1:]
        data['pred'] = KeepLargestConnectedComponent(applied_labels=applied_labels)(data['pred'])
    
    # eval infer tta
    if 'label' in data.keys():
        tta_dc_vals, tta_hd95_vals = eval_label_pred(data, args.out_channels, args.device)
        print('infer test time aug:')
        print('dice:', tta_dc_vals)
        print('hd95:', tta_hd95_vals)
        ret_dict['tta_dc'] = tta_dc_vals
        ret_dict['tta_hd'] = tta_hd95_vals
        
        # post label transform 
        sqz_transform = SqueezeDimd(keys=['label'])
        data = sqz_transform(data)
    
    # post transform
    data = post_transform(data)
    
    # eval infer origin
    if 'label' in data.keys():
        # get orginal label
        lbl_dict = {'label': data['label_meta_dict']['filename_or_obj']}
        label_loader = get_label_transform(args.data_name, keys=['label'])
        lbl_data = label_loader(lbl_dict)
        
        data['label'] = lbl_data['label']
        data['label_meta_dict'] = lbl_data['label']
        
        ori_dc_vals, ori_hd95_vals = eval_label_pred(data, args.out_channels, args.device)
        print('infer test original:')
        print('dice:', ori_dc_vals)
        print('hd95:', ori_hd95_vals)
        ret_dict['ori_dc'] = ori_dc_vals
        ret_dict['ori_hd'] = ori_hd95_vals
    
    if args.data_name == 'mmwhs':
        mmwhs_transform = Compose([
            LabelFilter(applied_labels=[1, 2, 3, 4, 5, 6, 7]),
            MapLabelValue(orig_labels=[0, 1, 2, 3, 4, 5, 6, 7],
                            target_labels=[0, 500, 600, 420, 550, 205, 820, 850]),
            # AddChannel(),
            # Spacing(
            #     pixdim=(args.space_x, args.space_y, args.space_z),
            #     mode=("nearest"),
            # ),
            # SqueezeDim()
        ])
        data['pred'] = mmwhs_transform(data['pred'])
        
    
    if not args.test_mode:
        # save pred result
        filename = get_filename(data)
        infer_img_pth = os.path.join(args.infer_dir, filename)

        save_img(
          data['pred'], 
          data['pred_meta_dict'], 
          infer_img_pth
        )
        
    return ret_dict

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import torch.nn.functional as F
from torch.cuda.amp import autocast

# 載入模型權重
def load_model_checkpoint(model, checkpoint_path, args):
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(args.device)
    model.eval()
    return model

def run_infering_with_gradcam(
        model,
        data,
        model_inferer,
        post_transform,
        args,
        target_layers
    ):
    ret_dict = {}
    device = args.device

    # 初始化 grad-cam
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())

    input_tensor = data['image'].to(device)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)

    # 可視化 Grad-CAM
    for i, cam_slice in enumerate(grayscale_cam):
        plt.imshow(cam_slice, cmap='jet')
        plt.title(f'Grad-CAM Slice {i}')
        plt.show()

    ret_dict['grayscale_cam'] = grayscale_cam

    
    # # test
    # start_time = time.time()
    # data['pred'] = infer(model, data, model_inferer, args.device)
    # end_time  = time.time()
    # ret_dict['inf_time'] = end_time-start_time
    # print(f'infer time: {ret_dict["inf_time"]} sec')
    
    # # post process transform
    # if args.infer_post_process:
    #     print('use post process infer')
    #     applied_labels = np.unique(data['pred'].flatten())[1:]
    #     data['pred'] = KeepLargestConnectedComponent(applied_labels=applied_labels)(data['pred'])
    
    # # eval infer tta
    # if 'label' in data.keys():
    #     tta_dc_vals, tta_hd95_vals = eval_label_pred(data, args.out_channels, args.device)
    #     print('infer test time aug:')
    #     print('dice:', tta_dc_vals)
    #     print('hd95:', tta_hd95_vals)
    #     ret_dict['tta_dc'] = tta_dc_vals
    #     ret_dict['tta_hd'] = tta_hd95_vals
        
    #     # post label transform 
    #     sqz_transform = SqueezeDimd(keys=['label'])
    #     data = sqz_transform(data)
    
    # # post transform
    # data = post_transform(data)
    
    # # eval infer origin
    # if 'label' in data.keys():
    #     # get orginal label
    #     lbl_dict = {'label': data['label_meta_dict']['filename_or_obj']}
    #     label_loader = get_label_transform(args.data_name, keys=['label'])
    #     lbl_data = label_loader(lbl_dict)
        
    #     data['label'] = lbl_data['label']
    #     data['label_meta_dict'] = lbl_data['label']
        
    #     ori_dc_vals, ori_hd95_vals = eval_label_pred(data, args.out_channels, args.device)
    #     print('infer test original:')
    #     print('dice:', ori_dc_vals)
    #     print('hd95:', ori_hd95_vals)
    #     ret_dict['ori_dc'] = ori_dc_vals
    #     ret_dict['ori_hd'] = ori_hd95_vals
    
    # if args.data_name == 'mmwhs':
    #     mmwhs_transform = Compose([
    #         LabelFilter(applied_labels=[1, 2, 3, 4, 5, 6, 7]),
    #         MapLabelValue(orig_labels=[0, 1, 2, 3, 4, 5, 6, 7],
    #                         target_labels=[0, 500, 600, 420, 550, 205, 820, 850]),
    #         # AddChannel(),
    #         # Spacing(
    #         #     pixdim=(args.space_x, args.space_y, args.space_z),
    #         #     mode=("nearest"),
    #         # ),
    #         # SqueezeDim()
    #     ])
    #     data['pred'] = mmwhs_transform(data['pred'])
        
    
    # if not args.test_mode:
    #     # save pred result
    #     filename = get_filename(data)
    #     infer_img_pth = os.path.join(args.infer_dir, filename)

    #     save_img(
    #       data['pred'], 
    #       data['pred_meta_dict'], 
    #       infer_img_pth
    #     )
        
    return ret_dict
