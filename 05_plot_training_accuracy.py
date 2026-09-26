"""Step 5: plot training accuracy by epoch from the step-4 training history.

Run 04_training.py first; it saves the history to training_history.json.
"""

import json
import sys

import matplotlib.pyplot as plt

history_path = "training_history.json"

try:
    with open(history_path) as f:
        history = json.load(f)
except FileNotFoundError:
    sys.exit(f"{history_path} not found. Run 04_training.py first.")

train_accuracy = history["train_metrics"]
epochs = range(1, len(train_accuracy) + 1)

plt.plot(epochs, train_accuracy, ".-")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("FashionMNIST Classifier: Training Accuracy by Epoch")
plt.xticks(epochs)
plt.grid()
plt.show()
