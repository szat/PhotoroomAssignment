import torch

# Check if CUDA (GPU support) is available in PyTorch
cuda_available = torch.cuda.is_available()

# Print whether CUDA is available or not
print("CUDA (GPU support) is available in PyTorch:", cuda_available)

# If CUDA is available, print out the name of the GPU
if cuda_available:
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    print("GPU Name:", gpu_name)

import cv2

# Test to see if OpenCV is working properly

# Attempt to load a version information from OpenCV to see if it's working
opencv_version = cv2.__version__

# Print the version to confirm it's working
print("OpenCV is working. Version:", opencv_version)