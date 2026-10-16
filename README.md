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

## Masking experiment (`mask_experiment.py`)

The dataset ships pre-made lung masks (`chest-xrays/mask/`, binary 0/255, one per
image), so no segmentation model is needed — masking is just an elementwise
multiply. `mask_experiment.py` trains the exact notebook model on raw vs masked
images (3 seeds, same 80/20 split) to test whether masking improves accuracy.

| condition                  | accuracy   | sens   | spec   | AUROC        |
| -------------------------- | ---------- | ------ | ------ | ------------ |
| raw                        | 0.8038     | 0.787  | 0.819  | **0.8557**   |
| masked (native scale)      | 0.5106     | 0.000  | 1.000  | 0.5000 (collapsed) |
| masked + per-image z-score | —          | —      | —      | 0.7651 (1 seed) |
| masked + lung-std rescale  | —          | —      | —      | 0.7113       |
| raw + per-image z-score    | —          | —      | —      | 0.8253       |

Findings:

- **Masking does not help — it costs ~0.1 AUROC** even with proper
  normalization (0.7113 vs 0.8253 standardized control, 0.8557 raw). Useful
  signal apparently exists outside the lung fields (ribs, mediastinum, overall
  chest geometry, possibly image-quality/marker cues).
- The masked model at the notebook's learning rate **collapsed entirely**
  (loss stuck at ln 2, predicts "no TB" for every image): masking drops the
  mean pixel value from 0.58 to 0.13 and, with no batch norm in this net, the
  ReLU activations fall below the point where gradients flow.
- This architecture is **brittle to input scale generally** — the raw model
  also collapsed when trained at lr 1e-2. Normalizing inputs (or adding batch
  norm) matters more than masking here.
- Bonus: evaluating with probabilities instead of hard predictions raises the
  raw AUROC from ~0.80 to 0.86.

Conclusion: keep the masks unused (they're still wired up as
`proc(imnumb, use_mask=True)` for future experiments). The bigger wins are
likely in the TODO list below (batch norm, proper batching, more capacity).

## TODO

- [x] Apply lung segmentation masks during preprocessing and compare accuracy → **tried, it hurts** (see above)
- [ ] Add metadata features (age, sex) as an extra input axis
- [ ] Batch training with a `DataLoader` instead of image-by-image
- [ ] Add batch norm — the net is currently fragile to input scale
- [ ] Add a validation set + early stopping
- [ ] Save/load the trained model
