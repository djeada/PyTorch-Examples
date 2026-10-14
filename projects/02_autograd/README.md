# 02 — Autograd and manual gradient descent


Calculate derivatives, inspect gradient accumulation, compute a vector-Jacobian
product, and fit `y = 3x + 2` by updating two scalar parameters manually.

## Read the source

1. Set `requires_grad=True` on a leaf tensor. Operations create a computation graph.
   For the scalar sum of squares, `backward()` computes the derivative `2x` using the chain rule.
2. Calling backward through a fresh graph adds to the same leaf's `.grad` buffer.
   Clearing gradients is a separate step from constructing the next graph.
3. For vector outputs, `torch.autograd.grad` takes `grad_outputs` and returns
   a vector-Jacobian product. The example computes gradients `[2, 7, 4]`;
   it does not construct the full Jacobian and does not accumulate into `.grad`.
4. The regression loop computes mean squared error, calls backward, and subtracts
   a learning-rate-scaled gradient from weight and bias.
5. Parameter updates run inside `torch.no_grad()` so the update itself is excluded
   from the next graph. Gradients are zeroed before another iteration.

## Try it

- Change the scalar function to `x.pow(3).sum()` and calculate its derivative by hand.
- Change the vector supplied as `grad_outputs` and derive the new product.
- Remove gradient clearing and inspect the effect on optimization.
- Change the learning rate and compare slow convergence with unstable updates.

## Numerical checks

Derivatives, accumulation and the vector-Jacobian product are checked against
analytical answers. Gradient descent must recover the known weight and bias.


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
