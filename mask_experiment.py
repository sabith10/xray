"""Compare raw vs lung-masked X-rays with the same model/hyperparameters.

Matches the notebook setup exactly (per-sample training, Adam lr=1e-3,
10 epochs, NLLLoss, 80/20 split with random_state=65) so the raw condition
is directly comparable to the committed AUROC of 0.7938.

Decoded 512x512 arrays are cached in cache/ to avoid re-decoding the
~3000x3000 source PNGs on every run.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix

FOLDER = 'chest-xrays'
SIZE = 512
EPOCHS = 10
LR = 1e-3
SEEDS = [0, 1, 2]


class ConvNeuralNet(nn.Module):
    """Same architecture as the notebook."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.pool = nn.MaxPool2d(16, stride=16)
        self.fc1 = nn.Linear(128, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.pool(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = torch.log_softmax(self.fc3(x), dim=1)
        return x


def load_data():
    """Decode all images once (raw + masked), cache to .npy."""
    os.makedirs('cache', exist_ok=True)
    raw_path = 'cache/raw.npy'
    masked_path = 'cache/masked.npy'
    labels_path = 'cache/labels.npy'

    metadata = pd.read_csv(f'{FOLDER}/MetaData.csv')

    if all(os.path.exists(p) for p in (raw_path, masked_path, labels_path)):
        x_raw = np.load(raw_path)
        x_masked = np.load(masked_path)
        y = np.load(labels_path)
        if len(y) == len(metadata):
            return x_raw, x_masked, y

    raw, masked, labels = [], [], []
    for _, row in metadata.iterrows():
        im = Image.open(f"{FOLDER}/images/{row['id']}.png").convert('L')
        imm = Image.open(f"{FOLDER}/mask/{row['id']}.png").convert('L')
        im = im.resize((SIZE, SIZE))
        imm = imm.resize((SIZE, SIZE), Image.NEAREST)  # NEAREST keeps mask binary

        img_array = (np.array(im) / 255.0).astype(np.float32)
        mask_array = np.array(imm) / 255.0

        raw.append(img_array)
        masked.append((img_array * mask_array).astype(np.float32))
        labels.append(row['ptb'])
        print(f'loaded {len(raw)}/{len(metadata)}', end='\r')

    x_raw = np.array(raw)
    x_masked = np.array(masked)
    y = np.array(labels)
    np.save(raw_path, x_raw)
    np.save(masked_path, x_masked)
    np.save(labels_path, y)
    print()
    return x_raw, x_masked, y


def train_and_eval(x_train, y_train, x_test, y_test, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    net = ConvNeuralNet().to(device)
    loss_function = nn.NLLLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i in range(len(x_train)):
            inp = torch.tensor(x_train[i]).unsqueeze(0).unsqueeze(0).to(device)
            label = torch.tensor([y_train[i]]).to(device)
            optimizer.zero_grad()
            loss = loss_function(net(inp), label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'    epoch {epoch+1}/{EPOCHS} loss: {running_loss/len(x_train):.3f}')

    # eval: probs for a proper AUROC, hard preds to match the notebook
    net.eval()
    probs, preds, all_labels = [], [], []
    with torch.no_grad():
        for i in range(len(x_test)):
            inp = torch.tensor(x_test[i]).unsqueeze(0).unsqueeze(0).to(device)
            out = net(inp)
            prob = out.exp()[0, 1].item()  # P(PTB)
            probs.append(prob)
            preds.append(out.argmax(1).item())
            all_labels.append(y_test[i])

    probs = np.array(probs)
    preds = np.array(preds)
    all_labels = np.array(all_labels)

    tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()
    return {
        'auroc_proba': roc_auc_score(all_labels, probs),
        'auroc_hard': roc_auc_score(all_labels, preds),  # notebook-style
        'sensitivity': tp / (tp + fn),
        'specificity': tn / (tn + fp),
        'accuracy': (preds == all_labels).mean(),
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    print('loading data...')
    x_raw, x_masked, y = load_data()
    print(f'dataset: {x_raw.shape}, positives: {y.sum()}/{len(y)}')

    idx = np.arange(len(y))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=65)

    results = {'raw': [], 'masked': []}
    for seed in SEEDS:
        for name, x in (('raw', x_raw), ('masked', x_masked)):
            print(f'[{name}] seed {seed}')
            r = train_and_eval(x[tr_idx], y[tr_idx], x[te_idx], y[te_idx],
                               seed, device)
            results[name].append(r)
            print(f'    acc {r["accuracy"]:.4f}  sens {r["sensitivity"]:.4f}  '
                  f'spec {r["specificity"]:.4f}  auroc {r["auroc_proba"]:.4f} '
                  f'(hard-pred {r["auroc_hard"]:.4f})')

    print('\n' + '=' * 70)
    print(f'mean over {len(SEEDS)} seeds (test n={len(te_idx)}):')
    print(f'{"condition":<10} {"accuracy":>9} {"sens":>7} {"spec":>7} '
          f'{"auroc":>7} {"auroc_hard":>10}')
    for name in ('raw', 'masked'):
        rs = results[name]
        m = {k: np.mean([r[k] for r in rs]) for k in rs[0]}
        s = {k: np.std([r[k] for r in rs]) for k in rs[0]}
        print(f'{name:<10} {m["accuracy"]:.4f}±{s["accuracy"]:.3f} '
              f'{m["sensitivity"]:.4f} {m["specificity"]:.4f} '
              f'{m["auroc_proba"]:.4f}±{s["auroc_proba"]:.3f} '
              f'{m["auroc_hard"]:.4f}')

    delta = (np.mean([r['auroc_proba'] for r in results['masked']])
             - np.mean([r['auroc_proba'] for r in results['raw']]))
    print(f'\nmasked - raw AUROC: {delta:+.4f}')


if __name__ == '__main__':
    main()
