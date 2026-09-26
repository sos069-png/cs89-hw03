"""Step 2: DataLoaders for the FashionMNIST train/valid/test sets.

Follows the "Building an Image Classifier with PyTorch" section of
10_neural_nets_with_pytorch.ipynb.
"""

import importlib

import torch
from torch.utils.data import DataLoader

# The step-1 module name starts with a digit, so it can't be imported with a
# plain `import` statement
data_setup = importlib.import_module("01_data_setup")
device = data_setup.device
train_and_valid_data = data_setup.train_and_valid_data
train_data = data_setup.train_data
valid_data = data_setup.valid_data
test_data = data_setup.test_data

# Only the training data is shuffled; validation and test order stays fixed
torch.manual_seed(42)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

if __name__ == "__main__":
    # Each entry is a tuple (image, target)
    X_sample, y_sample = train_data[0]
    print(f"Image shape: {tuple(X_sample.shape)}")  # [channels, rows, columns]
    print(f"Class: {train_and_valid_data.classes[y_sample]}")

    X_batch, y_batch = next(iter(train_loader))
    print(f"Batch shape: {tuple(X_batch.shape)}, labels: {tuple(y_batch.shape)}")
    print(f"Batches - train: {len(train_loader)}, valid: {len(valid_loader)}, "
          f"test: {len(test_loader)}")
