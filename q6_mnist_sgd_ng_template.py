"""Q6: MNIST classifier trained by SGD and a diagonal empirical-Fisher method.

The second branch is a computationally cheap approximation to natural-gradient
training. It keeps only diagonal empirical-Fisher information, so it should be
reported as an approximate method rather than exact NG.
"""
import copy
import csv
import gzip
import shutil
import struct
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / 'mnist_data'
OUT = ROOT / 'q6_outputs'
OUT.mkdir(exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH = 128
EPOCHS = 3
LR = 0.05
DAMPING = 1e-3
SEEDS = [11, 22, 33, 44, 55]

RAW = DATA_ROOT / 'MNIST' / 'raw'


def ensure_uncompressed(filename: str) -> Path:
    """Return an IDX file, decompressing filename.gz when necessary."""
    raw_path = RAW / filename
    if raw_path.exists():
        return raw_path
    gz_path = RAW / f'{filename}.gz'
    if gz_path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        with gzip.open(gz_path, 'rb') as source, raw_path.open('wb') as target:
            shutil.copyfileobj(source, target)
        return raw_path
    raise FileNotFoundError(
        f'Missing MNIST file: {raw_path} (or {gz_path}). '
        'Place the four standard MNIST raw IDX files in mnist_data/MNIST/raw.'
    )


def read_idx_images(filename: str) -> torch.Tensor:
    path = ensure_uncompressed(filename)
    with path.open('rb') as file:
        magic, count, rows, cols = struct.unpack('>IIII', file.read(16))
        if magic != 2051:
            raise ValueError(f'Unexpected image IDX magic number {magic} in {path}')
        array = np.frombuffer(file.read(), dtype=np.uint8).reshape(count, rows, cols).copy()
    return torch.from_numpy(array).float().unsqueeze(1) / 255.0


def read_idx_labels(filename: str) -> torch.Tensor:
    path = ensure_uncompressed(filename)
    with path.open('rb') as file:
        magic, count = struct.unpack('>II', file.read(8))
        if magic != 2049:
            raise ValueError(f'Unexpected label IDX magic number {magic} in {path}')
        array = np.frombuffer(file.read(), dtype=np.uint8).copy()
        if len(array) != count:
            raise ValueError(f'Label count mismatch in {path}')
    return torch.from_numpy(array).long()


train_ds = TensorDataset(
    read_idx_images('train-images-idx3-ubyte'),
    read_idx_labels('train-labels-idx1-ubyte'),
)
test_ds = TensorDataset(
    read_idx_images('t10k-images-idx3-ubyte'),
    read_idx_labels('t10k-labels-idx1-ubyte'),
)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.Tanh(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def make_train_loader(seed: int):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(train_ds, batch_size=BATCH, shuffle=True, generator=generator)


def evaluate(model):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / total


def train_sgd(model, train_loader):
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
        history.append(evaluate(model))
    return history


def train_diag_ng(model, train_loader):
    loss_fn = nn.CrossEntropyLoss()
    fisher_diagonal = [torch.zeros_like(parameter) for parameter in model.parameters()]
    beta = 0.95
    history = []
    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            model.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            with torch.no_grad():
                for parameter, fisher in zip(model.parameters(), fisher_diagonal):
                    if parameter.grad is None:
                        continue
                    # EMA of squared gradients: a diagonal empirical-Fisher proxy.
                    fisher.mul_(beta).addcmul_(parameter.grad, parameter.grad, value=1 - beta)
                    # RMS-scaled diagonal preconditioner; computationally close to RMSProp.
                    parameter.addcdiv_(parameter.grad, fisher.sqrt().add_(DAMPING), value=-LR)
        history.append(evaluate(model))
    return history


def run_method(method_name, seed):
    torch.manual_seed(seed)
    base_model = MLP().to(DEVICE)
    train_loader = make_train_loader(seed)
    start_time = time.time()
    if method_name == 'SGD':
        history = train_sgd(base_model, train_loader)
    elif method_name == 'diag-Fisher':
        history = train_diag_ng(base_model, train_loader)
    else:
        raise ValueError(f'Unknown method: {method_name}')
    elapsed = time.time() - start_time
    return history, elapsed


if __name__ == '__main__':
    print('device =', DEVICE)
    all_results = []
    histories = {'SGD': [], 'diag-Fisher': []}

    for seed in SEEDS:
        print(f'\nseed = {seed}')
        for method_name in ['SGD', 'diag-Fisher']:
            history, elapsed = run_method(method_name, seed)
            histories[method_name].append(history)
            row = {
                'seed': seed,
                'method': method_name,
                'epoch_1_accuracy': history[0],
                'epoch_2_accuracy': history[1],
                'epoch_3_accuracy': history[2],
                'seconds': elapsed,
            }
            all_results.append(row)
            print(method_name, 'history =', history, 'seconds =', elapsed)

    csv_path = OUT / 'q6_multiseed_results.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)

    print('\nsummary over seeds')
    summary_rows = []
    for method_name in ['SGD', 'diag-Fisher']:
        array = np.asarray(histories[method_name], dtype=float)
        mean_history = array.mean(axis=0)
        std_history = array.std(axis=0, ddof=1)
        elapsed_values = np.asarray([row['seconds'] for row in all_results if row['method'] == method_name])
        print(method_name)
        print('  mean accuracy by epoch =', mean_history)
        print('  std accuracy by epoch  =', std_history)
        print('  final accuracy mean/std =', mean_history[-1], std_history[-1])
        print('  seconds mean/std =', elapsed_values.mean(), elapsed_values.std(ddof=1))
        summary_rows.append((method_name, mean_history, std_history))

    fig, axis = plt.subplots(figsize=(7, 5))
    epochs = np.arange(1, EPOCHS + 1)
    for method_name, mean_history, std_history in summary_rows:
        axis.errorbar(epochs, mean_history, yerr=std_history, marker='o', capsize=4, label=method_name)
    axis.set_xlabel('epoch')
    axis.set_ylabel('test accuracy')
    axis.set_title('MNIST test accuracy across five random seeds')
    axis.set_xticks(epochs)
    axis.legend()
    fig.tight_layout()
    plot_path = OUT / 'q6_accuracy_multiseed.png'
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    print('\nsaved CSV to:', csv_path)
    print('saved accuracy plot to:', plot_path)
