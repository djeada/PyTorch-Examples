# 05 — Nonlinear classification with an MLP


Classify points on two concentric rings using a multilayer perceptron.
A single linear decision boundary cannot separate the inner and outer rings.

## Read the source

1. Generate independent training and test points from noisy ring radii and random angles.
2. Hidden linear layers transform the two input features. ReLU makes the composition
   nonlinear; stacking linear layers without nonlinear activations would still be linear.
3. The last layer returns two logits. `CrossEntropyLoss` combines the required
   log-softmax computation with negative log likelihood for integer labels.
4. Adam updates parameters after each mini-batch. The dataset, model and training
   loop live entirely in this folder.
5. `argmax(dim=1)` converts a batch of logits to class predictions for test accuracy.

## Try it

- Remove both ReLU layers and inspect the decision problem the model can represent.
- Change hidden-layer widths and compare parameter counts and accuracy.
- Make ring radii closer together or increase noise to make classification harder.
- Use softmax only when you want probabilities for display, keeping raw logits for the loss.

## Numerical checks

Loss must decrease by an order of magnitude, the output shape must match two
classes, and independent test accuracy must reach 95%.


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
