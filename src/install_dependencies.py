import os

# Uninstall conflicting versions
os.system("pip uninstall -y torch torchvision torchaudio tensorflow tensorflow-addons typeguard inflect diffusers transformers")

# Install PyTorch & CUDA dependencies
os.system("pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118")

# Install TensorFlow & Addons
os.system("pip install tensorflow==2.18.0 typeguard==2.13.3 tensorflow-addons==0.23.0")

# Install Diffusers, Transformers, and ONNX-related libraries
os.system("pip install diffusers transformers accelerate onnx onnx-tf")

# Verify installation
import torch
import torchvision
import tensorflow as tf

print(" PyTorch version:", torch.__version__)
print(" Torchvision version:", torchvision.__version__)
print(" TensorFlow version:", tf.__version__)

