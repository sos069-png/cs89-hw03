"""Step 4: training and evaluation functions for the image classifier.

Follows train2() and evaluate_tm() from 10_neural_nets_with_pytorch.ipynb,
as used in its "Building an Image Classifier with PyTorch" section.
"""

import importlib

import torch
import torch.nn as nn
import torchmetrics

# The step-2/3 module names start with a digit, so they can't be imported
# with a plain `import` statement
dataloaders = importlib.import_module("02_dataloaders")
model_module = importlib.import_module("03_model")
device = dataloaders.device
train_loader = dataloaders.train_loader
valid_loader = dataloaders.valid_loader
model = model_module.model


def evaluate_tm(model, data_loader, metric):
    model.eval()
    metric.reset()  # reset the metric at the beginning
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            metric.update(y_pred, y_batch)  # update it at each iteration
    return metric.compute()  # compute the final result at the end


def train2(model, optimizer, criterion, metric, train_loader, valid_loader,
           n_epochs):
    history = {"train_losses": [], "train_metrics": [], "valid_metrics": []}
    for epoch in range(n_epochs):
        total_loss = 0.
        metric.reset()
        for X_batch, y_batch in train_loader:
            model.train()
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            metric.update(y_pred, y_batch)
        mean_loss = total_loss / len(train_loader)
        history["train_losses"].append(mean_loss)
        history["train_metrics"].append(metric.compute().item())
        history["valid_metrics"].append(
            evaluate_tm(model, valid_loader, metric).item())
        print(f"Epoch {epoch + 1}/{n_epochs}, "
              f"train loss: {history['train_losses'][-1]:.4f}, "
              f"train accuracy: {history['train_metrics'][-1]:.4f}, "
              f"valid accuracy: {history['valid_metrics'][-1]:.4f}")
    return history


n_epochs = 20
xentropy = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)

if __name__ == "__main__":
    history = train2(model, optimizer, xentropy, accuracy, train_loader,
                     valid_loader, n_epochs)
