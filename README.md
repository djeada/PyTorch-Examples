# Learn PyTorch through seventeen independent projects

Start with tensors and automatic differentiation, build explicit training loops,
then explore image and sequence models, attention, model selection and checkpoints.
Continue with a tiny generative language model, graph and point-cloud learning,
physics-informed networks, neural ODEs and differentiable PDE simulation.
Each numbered folder is a complete standalone Python project with `main.py`,
`requirements.txt`, a source walkthrough, experiments and numerical checks.

All examples run on CPU and generate their data locally using PyTorch. No GPU,
notebook server, pretrained weights, downloaded datasets or shared training
framework is needed. Repeated training loops are intentional: forward passes,
gradient clearing, backward passes and optimizer steps remain visible in every lesson.

## The seventeen projects

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
| 11 | [A tiny causal language model](projects/11_minimal_language_model) | Character tokens, decoder blocks, causal attention, next-token loss and generation |
| 12 | [Graph message passing and permutation symmetry](projects/12_graph_neural_network) | Neighborhood aggregation, node equivariance and graph invariance |
| 13 | [Learning from unordered point clouds](projects/13_point_cloud_geometry) | Shared point features, symmetric pooling and geometric transformations |
| 14 | [A physics-informed harmonic oscillator](projects/14_physics_informed_ode) | ODE residuals, higher derivatives, hard initial conditions and conserved energy |
| 15 | [Learning a vector field through an ODE solver](projects/15_neural_ode) | Neural ODEs, differentiable RK4 and time-step refinement |
| 16 | [A physics-informed two-dimensional Poisson solver](projects/16_poisson_pde) | Laplacians, collocation, Dirichlet boundaries and manufactured-solution validation |
| 17 | [Differentiable heat flow and inverse parameter fitting](projects/17_differentiable_heat_pde) | PDE time stepping, diffusivity inference, stability and energy dissipation |

The teaching sequence builds up concepts, but the code has no dependencies
between lessons. Copy one project folder outside this repository and run it.
The generated problems make the mechanics inspectable and repeatable; their
reported accuracy and errors describe synthetic tasks, not real-world benchmarks.

Projects 01–10 teach the foundations. Project 11 introduces a small decoder-only
language model; 12–13 introduce geometric learning; 14–17 introduce scientific
machine learning. The scientific lessons compare against known equations and
reference solutions, and keep solver accuracy separate from learning accuracy.

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
`--epochs`, while the language model uses `--steps` for random-window updates.
Run `python path/to/main.py --help` for details. Defaults use seed 42,
one CPU thread and small training budgets. Reducing epochs can cause convergence
checks to fail, which is useful when exploring the optimization process.
Results are printed as JSON, including losses, shapes, accuracy or checkpoint checks.

## Language, geometry and scientific examples

```sh
python projects/11_minimal_language_model/main.py --prompt "the " --tokens 80
python projects/12_graph_neural_network/main.py --check
python projects/13_point_cloud_geometry/main.py --check
python projects/14_physics_informed_ode/main.py --check
python projects/15_neural_ode/main.py --check
python projects/16_poisson_pde/main.py --check
python projects/17_differentiable_heat_pde/main.py --check
```

The language model learns a tiny inline grammar; it demonstrates LLM mechanics
without pretrained weights or large-scale training. Geometric projects use core
PyTorch without requiring PyTorch Geometric. ODE/PDE examples use core autograd
and explicit numerical methods without requiring external solver packages.
Projects 14–17 use float64 for coordinate derivatives and numerical comparisons.

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
The extended checks also verify causal attention, graph and point permutation
symmetries, initial and boundary conditions, conserved or dissipated energy,
held-out ODE/PDE residuals, solver refinement and physical-parameter recovery.

GitHub Actions repeats formatting, syntax and independent-project checks on
CPU with Python 3.10 and 3.12. The runtime examples need only the regular
requirements; Ruff is a development tool.

## References

- [PyTorch basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
- [Autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
- [Optimization](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- [Saving and loading](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
- [Scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention)
- [Graph message passing](https://arxiv.org/abs/1810.00826)
- [PointNet](https://arxiv.org/abs/1612.00593)
- [Physics-informed neural networks](https://arxiv.org/abs/1711.10561)
- [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366)

## License

The existing MIT license is retained; see [LICENSE](LICENSE).
