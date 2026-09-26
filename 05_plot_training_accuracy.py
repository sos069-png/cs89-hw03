# Step 5: plot training accuracy by epoch from the training history

import matplotlib.pyplot as plt

train_accuracy = history["train_metrics"]
epochs = range(1, len(train_accuracy) + 1)

plt.plot(epochs, train_accuracy, ".-")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("FashionMNIST Classifier: Training Accuracy by Epoch")
plt.xticks(epochs)
plt.grid()
plt.show()
