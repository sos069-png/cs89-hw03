# Step 1: imports, device selection, and FashionMNIST dataset setup
# (follows "Building an Image Classifier with PyTorch" in
# 10_neural_nets_with_pytorch.ipynb)

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as T

# Select the best available device: CUDA GPU, Apple Silicon GPU (MPS), or CPU
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Convert PIL images to float32 tensors with pixel values scaled to [0, 1]
toTensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])

train_and_valid_data = torchvision.datasets.FashionMNIST(
    root="datasets", train=True, download=True, transform=toTensor)
test_data = torchvision.datasets.FashionMNIST(
    root="datasets", train=False, download=True, transform=toTensor)

# Split the original 60,000 training images into 55,000 train / 5,000 valid
torch.manual_seed(42)
train_data, valid_data = torch.utils.data.random_split(
    train_and_valid_data, [55_000, 5_000])

print(f"Using device: {device}")
print(f"Train: {len(train_data)}, valid: {len(valid_data)}, "
      f"test: {len(test_data)}")
