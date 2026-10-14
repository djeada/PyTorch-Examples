# 15 — Learning a vector field through an ODE solver


Learn an autonomous vector field from observed trajectories by differentiating
through a fourth-order Runge-Kutta integrator.

$$\frac{dy}{dt}=f_\theta(y).$$

Training trajectories come from the known test system `y' = -0.7*y`. The network
receives states, and its derivative predictions are integrated to produce trajectories.

## Read the source

1. A small MLP represents the vector field. Unlike project 14's coordinate-to-solution
   network, it maps the current state to its time derivative.
2. Each RK4 step evaluates four slopes and combines them to update the state.
   All stages are ordinary tensor operations with no detach between steps.
3. Backpropagation through the unrolled solver updates vector-field parameters
   from trajectory error. A nonzero parameter gradient checks that this path exists.
4. Test with unseen initial states and a slightly longer time interval.
   Compare both trajectories and vector-field values with the analytic test system.
5. Halve the time step and compare aligned trajectory samples to inspect numerical sensitivity.

This is fixed-step discrete backpropagation through RK4. Its memory grows with
the unrolled trajectory; it does not implement an adjoint method or adaptive solver.
There is no `torchdiffeq` dependency. Float64 makes the refinement comparison useful.

## Try it

- Replace RK4 with Euler integration and compare step-size effects.
- Learn a two-dimensional vector field, such as a damped oscillator.
- Increase the forecast horizon and separate integration error from learned-field error.
- Detach an intermediate state deliberately and inspect the changed gradient path.

## Numerical checks

Gradients must reach the vector field, loss must decrease, new trajectories must
match the reference, and time-step refinement must produce consistent outputs.

## Reference

[Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366)


## Run independently

This folder contains its own `main.py`, `requirements.txt` and teaching README.
Use Python 3.10+ and the same CPU environment as the [course setup](../../README.md).
It can be copied outside the repository and needs no other lesson's source.

```sh
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python main.py
python main.py --check
```

On macOS, install the native wheel with `python -m pip install -r requirements.txt`.
Run `python main.py --help` for options. Defaults use seed 42, one CPU thread,
and a small example-specific training budget. `--epochs` changes that budget;
shorter runs can fail convergence checks. Results are printed as JSON.
