"""Step 3: the FashionMNIST image classifier (an MLP).

Follows the "Building an Image Classifier with PyTorch" section of
10_neural_nets_with_pytorch.ipynb.
"""

import importlib

import torch
import torch.nn as nn

# The step-1 module name starts with a digit, so it can't be imported with a
# plain `import` statement
device = importlib.import_module("01_data_setup").device


class ImageClassifier(nn.Module):
    def __init__(self, n_inputs, n_hidden1, n_hidden2, n_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_inputs, n_hidden1),
            nn.ReLU(),
            nn.Linear(n_hidden1, n_hidden2),
            nn.ReLU(),
            nn.Linear(n_hidden2, n_classes)
        )

    def forward(self, X):
        return self.mlp(X)


torch.manual_seed(42)
model = ImageClassifier(n_inputs=1 * 28 * 28, n_hidden1=300, n_hidden2=100,
                        n_classes=10).to(device)

if __name__ == "__main__":
    print(model)
    n_params = sum([param.numel() for param in model.parameters()])
    print(f"Parameters: {n_params:,}")

    # Run one validation batch through the untrained model to check shapes
    valid_loader = importlib.import_module("02_dataloaders").valid_loader
    X_batch, _ = next(iter(valid_loader))
    model.eval()
    with torch.no_grad():
        y_pred_logits = model(X_batch.to(device))
    print(f"Input batch: {tuple(X_batch.shape)} -> "
          f"logits: {tuple(y_pred_logits.shape)}")
