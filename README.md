# Learn PyTorch through ten independent projects

Start with tensors and automatic differentiation, build explicit training loops,
then explore image and sequence models, attention, model selection and checkpoints.
Each numbered folder is a complete standalone Python project with `main.py`,
`requirements.txt`, a source walkthrough, experiments and numerical checks.

All examples run on CPU and generate their data locally using PyTorch. No GPU,
notebook server, pretrained weights, downloaded datasets or shared training
framework is needed. Repeated training loops are intentional: forward passes,
gradient clearing, backward passes and optimizer steps remain visible in every lesson.

## The ten projects

| Project | Lesson | Concepts |
| --- | --- | --- |
| 01 | [Tensors, shapes and broadcasting](projects/01_tensors) | Shapes, dtype, device, broadcasting, matrix multiplication and storage |
| 02 | [Autograd and manual gradient descent](projects/02_autograd) | Chain rule, accumulation, vector-Jacobian products and manual updates |
| 03 | [Modules, losses and optimizers](projects/03_linear_regression) | nn.Module, parameters, MSE and SGD |
| 04 | [Datasets and mini-batches](projects/04_datasets_and_dataloaders) | Custom Dataset, DataLoader, shuffling and partial batches |
| 05 | [Nonlinear classification with an MLP](projects/05_multilayer_networks) | Hidden layers, nonlinear activations, logits and Adam |
| 06 | [Image classification with a CNN](projects/06_convolutional_networks) | Image layout, convolutions, pooling and held-out accuracy |
| 07 | [Dropout, validation and early stopping](projects/07_regularization_and_validation) | Train/eval modes, dropout, AdamW, validation and early stopping |
| 08 | [Sequence forecasting with a GRU](projects/08_sequence_forecasting) | Sequence shapes, recurrent state and a forecasting baseline |
| 09 | [Embeddings, attention and padding masks](projects/09_attention_and_transformers) | Token/position embeddings, attention and correct padding masks |
| 10 | [Saving, loading, inference and resuming training](projects/10_checkpoints_and_inference) | state_dict, optimizer/RNG state, inference and reproducible resumption |

The teaching sequence builds up concepts, but the code has no dependencies
between lessons. Copy one project folder outside this repository and run it.
The generated problems make the mechanics inspectable and repeatable; their
reported accuracy and errors describe synthetic tasks, not real-world benchmarks.

## Setup

Use Python 3.10 or newer. The requirements pin PyTorch 2.14.1 and include NumPy
for PyTorch's NumPy integration. The examples themselves use PyTorch tensors.

Create a virtual environment from the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.
Install the CPU wheel on Linux or Windows, then the remaining requirements:

```sh
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

On macOS, use `python -m pip install -r requirements.txt` to install the native
wheel. Refer to the [official installer](https://pytorch.org/get-started/locally/)
for supported platform builds. The examples create CPU tensors explicitly through
ordinary CPU-default operations and do not select an accelerator.

## Start with the foundations

```sh
python projects/01_tensors/main.py
python projects/02_autograd/main.py
python projects/03_linear_regression/main.py
```

Use `--check` to verify a lesson's expected numerical behavior:

```sh
python projects/02_autograd/main.py --check
python projects/06_convolutional_networks/main.py --check
```

Every lesson supports `--seed` and `--threads`; training lessons also support
`--epochs`. Run `python path/to/main.py --help` for details. Defaults use seed 42,
one CPU thread and small training budgets. Reducing epochs can cause convergence
checks to fail, which is useful when exploring the optimization process.
Results are printed as JSON, including losses, shapes, accuracy or checkpoint checks.

Project 10 saves to `outputs/checkpoint.pt` on ordinary runs. Use `--output` to
choose another path. Its check mode uses a temporary checkpoint by default.

## Check the complete course

```sh
python -m pip install -r requirements-dev.txt
ruff check .
ruff format --check .
python scripts/check_projects.py
```

The runner copies each project into its own temporary directory outside the
repository and executes `main.py --check` with Python's isolated `-I` mode.
This excludes the repository and `PYTHONPATH` from import resolution. Only the
installed dependencies and the copied project files are available. It runs
both seeds 42 and 7 by default; `--seeds 42` selects a single-seed check.

Checks cover analytical gradients and tensor results, decreasing training loss,
held-out metrics, final partial batches, dropout evaluation behavior, forecasting
against a baseline, padding invariance, checkpoint prediction equality and exact
resumption of the next optimizer step. Every child process has a timeout.

GitHub Actions repeats formatting, syntax and independent-project checks on
CPU with Python 3.10 and 3.12. The runtime examples need only the regular
requirements; Ruff is a development tool.

## References

- [PyTorch basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
- [Autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
- [Optimization](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- [Saving and loading](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)

## License

The existing MIT license is retained; see [LICENSE](LICENSE).
