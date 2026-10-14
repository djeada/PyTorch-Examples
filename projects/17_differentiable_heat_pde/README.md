# 17 — Differentiable heat flow and inverse parameter fitting


Infer heat diffusivity from sparse observations by backpropagating through a
finite-difference time-stepping solver.

$$u_t=\kappa u_{xx},\quad u(0,t)=u(1,t)=0,\quad u(x,0)=\sin(\pi x).$$

The continuous reference is `exp(-kappa*pi**2*t)*sin(pi*x)`. Observations are
generated from that analytic solution, not from the solver used for training.

## Read the source

1. Use a centered second difference in space and forward Euler in time.
   The interior update is `u + r*(u_right - 2*u + u_left)`, with `r=kappa*dt/dx**2`.
2. For this one-dimensional explicit scheme, stability requires `r <= 0.5`.
   A bounded sigmoid parameterization keeps diffusivity positive and within a
   range that satisfies the restriction for the chosen grid and time step.
3. Set boundary values to zero at every step. Keep all steps in the computation
   graph so sparse observation error differentiates back to the diffusivity parameter.
4. Fit using four sensors at four observation times, then check the full field.
   Small discretization bias means the recovered coefficient need not equal
   the continuum coefficient to every digit.
5. Check dissipation of the discrete squared-amplitude energy, zero boundaries,
   and a spatial/time refinement comparison.

This is differentiable numerical simulation with one learned physical parameter,
not a neural-network solution ansatz. It complements the PINN in project 16.
The small float64 solver keeps its stability condition and gradient path visible.

## Try it

- Change spatial resolution and adjust the time step to preserve stability.
- Add observation noise and inspect uncertainty in the fitted coefficient.
- Use a different initial profile and compare the dissipation behavior.
- Replace explicit Euler with an implicit step and contrast stability and solve cost.

## Numerical checks

Parameter gradients, diffusivity recovery, observation fit, boundary values,
energy decay, analytic field accuracy and grid refinement are all verified.


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
