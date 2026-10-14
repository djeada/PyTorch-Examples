# 07 — Dropout, validation and early stopping


Fit a noisy sine curve using dropout, weight decay, validation-based model selection
and early stopping. Keep the test set untouched until after selecting a model.

## Read the source

1. Generate training, validation and test data independently with separate seeds.
2. `Dropout` randomly removes activations during `train()` and is disabled in `eval()`.
   AdamW applies decoupled weight decay in addition to the gradient-based update.
3. Check validation MSE after each epoch under `eval()` and `no_grad()`.
   Evaluation mode alone does not disable gradient tracking.
4. Save the best weights with `deepcopy(model.state_dict())`. A plain state dictionary
   contains tensor references that can change as training continues.
5. Stop after 60 epochs without sufficient improvement and restore the best weights.
   Select using validation loss; calculate test loss only after that selection.

## Try it

- Remove dropout and weight decay separately, comparing train and validation MSE.
- Increase network size while keeping the small training dataset fixed.
- Change patience or the minimum improvement threshold.
- Compare repeated predictions in training and evaluation modes.

## Numerical checks

Validation loss must improve, test MSE must stay small, and repeated evaluation
predictions must be identical with dropout disabled.


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
