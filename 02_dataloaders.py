# Step 2: DataLoaders for the FashionMNIST train/valid/test sets

from torch.utils.data import DataLoader

# Only the training data is shuffled; validation and test order stays fixed
torch.manual_seed(42)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

# Each entry is a tuple (image, target)
X_sample, y_sample = train_data[0]
print(f"Image shape: {tuple(X_sample.shape)}")  # [channels, rows, columns]
print(f"Class: {train_and_valid_data.classes[y_sample]}")
