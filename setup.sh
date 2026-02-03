#!/bin/bash


# sudo apt-get update
# sudo apt-get install -y python3-dev build-essential ninja-build
# sudo apt-get install ffmpeg libsm6 libxext6  -y

# #1. Install compilers and build tools (replaces build-essential, python3-dev, ninja-build)
# conda install -n grit -y -c conda-forge gcc_linux-64 gxx_linux-64 make ninja

# # 2. Install FFmpeg and X11 libraries (replaces ffmpeg, libsm6, libxext6)
# conda install -n grit -y -c conda-forge ffmpeg xorg-libxext xorg-libsm

# conda activate grit

#1. Install compilers and build tools (replaces build-essential, python3-dev, ninja-build)
conda install -y -c conda-forge gcc_linux-64 gxx_linux-64 make ninja

# 2. Install FFmpeg and X11 libraries (replaces ffmpeg, libsm6, libxext6)
conda install -y -c conda-forge ffmpeg xorg-libxext xorg-libsm

#conda activate grit

cd trl
pip install -e .

pip install torch==2.5.1 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

pip install git+https://github.com/huggingface/transformers.git@8ee50537fe7613b87881cd043a85971c85e99519
pip install git+https://github.com/huggingface/accelerate.git

# CUDA_HOME=/usr/local/cuda 
pip install deepspeed --no-deps
pip install mpi4py

pip install qwen-vl-utils
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
pip install trl
pip install nltk
pip install fastapi
pip install python-multipart
pip install uvicorn
pip install peft
# pip install accelerate
pip install datasets
pip install bitsandbytes
pip install flash-attn==2.7.4.post1
pip install wandb

pip install openai
pip install distro
pip install httpx
pip install anyio
pip install typing-extensions

pip install latex2sympy2-extended
pip install math_verify
pip install jsonlines

pip install Pillow
pip install "numpy<2.0.0"

pip install regex
pip install deepspeed
pip install decord timm opencv-python imageio


echo "All dependencies installed successfully!"
