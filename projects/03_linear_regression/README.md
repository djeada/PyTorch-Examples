# 03 — Modules, losses and optimizers


Fit noisy samples of `y = 3x + 1` with a trainable `nn.Module` and an optimizer.
Training and test samples are generated independently.

## Read the source

1. `nn.Linear(1, 1)` owns a weight and bias registered as model parameters.
   Subclassing `nn.Module` and defining `forward` makes the computation explicit.
2. Inputs and targets both have shape `(samples, 1)`. A mismatched target shape
   can broadcast and silently define a different regression problem.
3. `MSELoss` measures the average squared prediction error. SGD receives
   `model.parameters()` so it knows which tensors to update.
4. Each iteration clears old gradients, runs the forward pass, computes loss,
   calls `backward`, then calls `optimizer.step`.
5. `eval()` selects evaluation behavior. `no_grad()` separately disables gradient
   tracking during evaluation; these calls solve different problems.

## Try it

- Change the true slope, intercept or noise level.
- Compare SGD with Adam using the same generated samples.
- Inspect the gradients before `optimizer.step()` and the parameters afterward.
- Compare training and test MSE rather than reporting training loss alone.

## Numerical checks

Training loss must decrease, the fitted parameters must approximate the known
line, and MSE must stay small on independent test samples.


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
