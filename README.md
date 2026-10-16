# xray — Chest X-ray TB Classification

A CNN built with PyTorch that classifies chest X-rays as tuberculosis-positive or
negative (`ptb` label from `chest-xrays/MetaData.csv`).

## Dataset

- 704 grayscale chest X-rays, resized to 512×512, normalized to `[0, 1]`
- Lung masks available in `chest-xrays/mask/` (currently **not used** — see TODO)
- 80/20 train/test split (`random_state=65`) → 563 train / 141 test images

## Model

```
ConvNeuralNet(
  (conv1): Conv2d(1, 64, kernel_size=(3, 3))
  (conv2): Conv2d(64, 128, kernel_size=(3, 3))
  (pool): MaxPool2d(kernel_size=16, stride=16)
  (fc1): Linear(128, 120)
  (fc2): Linear(120, 84)
  (fc3): Linear(84, 2)
)
```

Training: `NLLLoss` + Adam (`lr=0.001`), 10 epochs, `log_softmax` output.

## Results

| Metric       | Value    |
| ------------ | -------- |
| **AUROC**    | **0.7938** |
| Sensitivity  | 0.7681   |
| Specificity  | 0.8194   |
| Accuracy     | 79.43%   |

## TODO

- [ ] Apply lung segmentation masks during preprocessing (`proc()` has the code commented out) and compare accuracy
- [ ] Add metadata features (age, sex) as an extra input axis
- [ ] Batch training with a `DataLoader` instead of image-by-image
- [ ] Add a validation set + early stopping
- [ ] Save/load the trained model
