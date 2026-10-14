# 01 — Tensors, shapes and broadcasting


Inspect a two-dimensional tensor, broadcast a vector across its rows, multiply
matrices, and observe the difference between shared and copied storage.

## Read the source

1. `torch.tensor` creates floating-point data with an explicit two-by-three shape.
   Every operation needs compatible shapes, dtypes and devices.
2. A length-three vector broadcasts across the two rows when added to the matrix.
   Broadcasting aligns trailing dimensions; equal dimensions or a dimension of one can expand.
3. `@` means matrix multiplication, while `*` means elementwise multiplication.
   Here `(2, 3) @ (3, 2)` produces `(2, 2)`.
4. `view` changes the metadata of this contiguous tensor while sharing storage.
   Writing through the view changes the original. `clone` allocates an independent copy.
   `reshape` may return a view or make a copy, depending on the layout.
5. The script prints shapes, dtype, device and numerical results as JSON.

## Try it

- Replace the broadcast vector with a `(2, 1)` tensor and predict the result.
- Compare `matrix * matrix` and `matrix @ matrix.T`.
- Transpose a tensor, inspect `is_contiguous()`, and compare `view`, `reshape` and `contiguous()`.
- Change `float32` to `float64` and inspect dtype propagation.

## Numerical checks

Exact expected broadcasting and matrix products, shared-storage mutation,
independent cloning, and CPU/float32 placement are verified.


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
