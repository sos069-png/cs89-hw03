# Assignment summary: FashionMNIST image classifier in PyTorch

## Approach
We built the classifier one script at a time, using the "Building an Image Classifier with PyTorch" section of `10_neural_nets_with_pytorch.ipynb` as the reference. Each step was run to check it before being committed. Later scripts reuse earlier ones through `importlib`, because Python can't `import` a file whose name starts with a digit. Each script's checks run only when it is run directly, not when another script imports it.

## Scripts
1. **`01_data_setup.py`**: imports and device selection (CUDA, then MPS, then CPU). Loads FashionMNIST, converting images to `float32` tensors scaled to [0, 1]. With seed 42, splits the 60,000 training images into 55,000 for training and 5,000 for validation; the test set has 10,000.
2. **`02_dataloaders.py`**: training, validation and test DataLoaders with batch size 32; only the training data is shuffled. A check prints one image's shape `(1, 28, 28)` and its class, "Ankle boot".
3. **`03_model.py`**: `ImageClassifier(nn.Module)` flattens each image, then applies Linear 784→300, ReLU, 300→100, ReLU, 100→10. Created with seed 42; it has 266,610 parameters, the same count the notebook gives.
4. **`04_training.py`**: `train2()` and `evaluate_tm()` ported from the notebook. Records training loss, training accuracy and validation accuracy for every epoch. Uses `CrossEntropyLoss`, SGD with lr = 0.1, and torchmetrics multiclass accuracy, for 20 epochs. Result: training accuracy 0.7814 → 0.9286; validation accuracy peaked at 0.8886 in epoch 16 and ended at 0.8788.
5. **`05_plot_training_accuracy.py`**: plots training accuracy by epoch (see below).
6. **`06_predictions.py`**: loads the trained weights and predicts the first three images of a validation batch. It converts the predicted indices to class names, compares them with the true labels, and prints softmax probabilities. All three were correct: Sneaker, Coat and Pullover. Pullover was the least certain at p = 0.592, with Coat and Shirt as the runners-up.

## Errors and corrections
- **Reference notebook missing:** the repository started empty, so the public version of the notebook was used at first. After the notebook was uploaded, its classifier section was confirmed to match.
- **Missing packages:** `torch`, `torchvision`, `torchmetrics` and `matplotlib` weren't installed. Installing PyTorch from its CPU-only package index failed with "No matching distribution found"; installing from the default package index (PyPI) worked.
- **Too much in step 1:** step 1 first included the DataLoaders, which went beyond what that step asked for. When step 2 was requested, they were moved there.
- **Training results not kept:** the history and the trained model were lost when `04_training.py` finished. It was changed to save them to `training_history.json` and `model_weights.pt`, both git-ignored, so steps 5 and 6 don't need to retrain.
- **Hard-to-read output:** the softmax probability tensor wrapped across lines, so it was changed to a table of classes by images.
- **No display in the environment:** the cloud environment has no screen, so the plot was checked by saving it to an image file.

## Training accuracy plot (step 5)
`05_plot_training_accuracy.py` reads `history["train_metrics"]` from `training_history.json` and plots it against epochs 1 to 20. The x-axis is labeled "Epoch", the y-axis "Accuracy", and the title is "FashionMNIST Classifier: Training Accuracy by Epoch". It adds a grid and a tick for every epoch, then calls `plt.show()`. If the history file is missing, it stops with a message to run `04_training.py` first. The curve rises steadily from 0.78 to 0.93. Validation accuracy leveled off around 0.88, which suggests mild overfitting in later epochs.

## How to run
Run `python3 04_training.py`, then `05_plot_training_accuracy.py` and `06_predictions.py`. Required packages: `torch`, `torchvision`, `torchmetrics` and `matplotlib`.
