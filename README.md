# Group 1 — CNN Design Challenge

## Assignment Overview
Custom CNN built from scratch to classify 15-class Tiny ImageNet images (64×64 RGB).
Trained on 5775 images, validated on 825. Evaluated on a hidden 20% test set by the instructor.

---

## Submission Files
| File | Description |
|------|-------------|
| `Group1_Assignment.ipynb` | Full pipeline: data loading, model definition, training, evaluation |
| `model.pth` | Saved weights from the best validation checkpoint |

---

## Model Architecture — `TinyImageNetCNN`
Sequential CNN with 5 convolutional blocks + Global Average Pooling:

| Block | Spatial Size | Channels | Notes |
|-------|-------------|----------|-------|
| Block 1 | 64→32 | 3→32 | 2× Conv-BN-ReLU, MaxPool, Dropout2d(0.1) |
| Block 2 | 32→16 | 32→64 | 2× Conv-BN-ReLU, MaxPool, Dropout2d(0.1) |
| Block 3 | 16→8 | 64→128 | 3× Conv-BN-ReLU, MaxPool, Dropout2d(0.2) |
| Block 4 | 8→4 | 128→256 | 3× Conv-BN-ReLU, MaxPool, Dropout2d(0.2) |
| Block 5 | 4→4 | 256→512 | 3× Conv-BN-ReLU, Dropout2d(0.3), no pool |
| GAP | 4→1 | 512 | AdaptiveAvgPool2d |
| Classifier | — | 512→256→15 | Linear-BN-ReLU-Dropout(0.4)-Linear |

**No skip connections, attention, or pre-trained weights used.**

---

## Training Details
- **Framework:** PyTorch
- **Optimizer:** AdamW (lr=3e-3, weight_decay=1e-4)
- **Scheduler:** OneCycleLR (cosine anneal, 30% warmup)
- **Loss:** CrossEntropyLoss with label smoothing (0.1)
- **Augmentation:** RandomHorizontalFlip, RandomCrop (pad=8), ColorJitter, RandomGrayscale, RandomErasing, Mixup (alpha=0.4)
- **Early stopping:** patience=20
- **Max epochs:** 150
- **Seed:** 42

---

## How to Load & Predict

```python
# Run the notebook cells up to the model definition, then:

# Load model
model = load_model('model.pth', num_classes=15)

# Predict — images_numpy: np.ndarray shape (N, 64, 64, 3) uint8
predictions = predict(images_numpy, model_path='model.pth', n_tta=8)
# Returns: np.ndarray shape (N,)  int64  class indices 0-14
```

---

## Dataset Setup
Place the data files in the `data/` directory:
```
data/
  train-70_.pkl
  validation-10_.pkl
```

---

## Inference Notes
- **Test-Time Augmentation (TTA x8):** 8 augmented passes averaged at inference for +2-3% accuracy boost.
- To disable TTA: pass `n_tta=1` to `predict()`.
- The `predict()` function auto-detects GPU/CPU.
