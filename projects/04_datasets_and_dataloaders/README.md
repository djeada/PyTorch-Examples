# 04 — Datasets and mini-batches


Classify three separated clusters with a custom dataset and shuffled mini-batches.

## Read the source

1. A `Dataset` exposes `__len__` and `__getitem__`; each item is a feature tensor
   and an integer class label. Data generation belongs to the dataset, not the model.
2. `DataLoader` groups samples into batches and shuffles their order for training.
   A seeded generator makes the shuffle reproducible. `num_workers=0` keeps this
   small example easy to run across platforms.
3. A linear classifier returns three raw logits. `CrossEntropyLoss` takes those
   logits and `torch.long` labels; no softmax is applied before the loss.
4. There are 150 samples and a batch size of 32, so the last batch has 22 samples.
   Weight each batch loss by its sample count when reporting a whole-epoch mean.
5. Evaluate on a separately generated test dataset without changing model weights.

## Try it

- Change batch size and predict the batch count and final batch size.
- Compare shuffled and fixed training order.
- Add a fourth class, updating the dataset and output layer together.
- Return an additional item such as a sample ID and inspect how the loader collates it.

## Numerical checks

Every training sample must be visited, the partial batch must be retained,
loss must improve, and held-out classification accuracy must reach 95%.


## Run independently

Copy this folder anywhere. It contains its own source and dependency file;
no repository package, other lesson, dataset download or GPU is required.
Use Python 3.10+ and install the dependencies in a virtual environment.

From this folder on Linux or Windows with a CPU PyTorch installation:

```sh
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python main.py
python main.py --check
```

On macOS, install with `python -m pip install -r requirements.txt` using the
native PyTorch wheel. See the [course setup](../../README.md) for virtual
environment commands. The folder remains runnable if copied without that README.

Run `python main.py --help` for options. The default seed is 42 and the default
CPU thread count is one. Training lessons expose `--epochs`; the numerical
checks target their default training budgets. Every run reports results as JSON.
