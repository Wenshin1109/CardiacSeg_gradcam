sudo apt update && \
sudo apt install ffmpeg libsm6 libxext6  -y && \
# pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113 && \
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121 && \
pip install "monai[nibabel, tqdm, einops]==1.2.0" && \
# pip install monailabel==0.7.0 && \
pip install monailabel==0.8.5 && \
pip install timm==0.6.13 && \
pip install tensorboard==2.13.0 && \
pip install ml-collections==0.1.1 && \
pip install scikit-learn==1.2.2 && \
pip install pandas==1.5.3 && \
pip install matplotlib==3.7.1 && \
pip install ray==2.4.0 && \
pip install torchsummaryX==1.3.0 && \
pip install toml==0.10.2 && \
pip install gdown==4.7.1 && \
pip install --user ipykernel && \
pip install grad-cam==1.5.4 && \
pip install opencv-python==4.10.0.84 && \
python -m ipykernel install --user --name base
