# 08 — Sequence forecasting with a GRU


Predict the next value after twelve observations of a noisy sine wave using a GRU.
Compare its error with predicting that the last observed value will repeat.

## Read the source

1. Each independently generated wave has a random phase and frequency.
   Training and test waves do not share observations or overlapping windows.
2. Inputs have shape `(batch, time, features)`, or `(256, 12, 1)` here.
   `batch_first=True` tells the GRU to interpret that layout.
3. A GRU passes a hidden state through the sequence, using trainable gates to
   control how information is retained and updated.
4. Take the output at the final input step and map its 24 features to one forecast.
   Both prediction and target have shape `(batch, 1)` for MSE.
5. Evaluate on new waves and compare against the last-value persistence baseline.

For real time series, split chronologically before creating overlapping windows,
and keep future observations out of training preprocessing and inputs.

## Try it

- Change the sequence length, updating data generation and slicing together.
- Replace the GRU with an LSTM and inspect its additional cell state.
- Predict several future steps and change the output layer and targets together.
- Increase noise and compare how much improvement remains over the simple baseline.

## Numerical checks

Input and prediction shapes must agree with the task, loss must decrease,
and held-out MSE must beat the persistence baseline by at least a factor of two.


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
