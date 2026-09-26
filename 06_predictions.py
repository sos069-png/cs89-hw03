# Step 6: predict classes for the first three images of a validation batch

import torch.nn.functional as F

classes = train_and_valid_data.classes

model.eval()
X_new, y_new = next(iter(valid_loader))
X_new, y_new = X_new[:3].to(device), y_new[:3]
with torch.no_grad():
    y_pred_logits = model(X_new)
y_pred = y_pred_logits.argmax(dim=1).cpu()  # index of the largest logit

y_proba = F.softmax(y_pred_logits, dim=1).cpu()

print("Predictions for the first 3 validation images:")
for i, (pred, true) in enumerate(zip(y_pred, y_new)):
    result = "correct" if pred == true else "WRONG"
    print(f"  Image {i}: predicted {classes[pred]!r} "
          f"(p={y_proba[i, pred]:.3f}), true {classes[true]!r} -> {result}")

print(f"\nPredicted indices: {y_pred.tolist()}")
print(f"True indices:      {y_new.tolist()}")

print("\nSoftmax probabilities:")
print(f"  {'Class':<14}" + "".join(f"{f'Image {i}':>10}"
                                   for i in range(len(y_proba))))
for index, name in enumerate(classes):
    print(f"  {name:<14}" + "".join(f"{p:>10.3f}" for p in y_proba[:, index]))
