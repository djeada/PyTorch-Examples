# 14 — A physics-informed harmonic oscillator


Solve a unit-frequency harmonic oscillator using its differential equation
and initial conditions, without training on trajectory labels.

$$u''(t) + u(t) = 0,\qquad u(0)=1,\quad u'(0)=0,\quad 0\leq t\leq 2.$$

## Read the source

1. The trial solution `1 + t**2 * network(t)` satisfies both initial conditions
   exactly, for every set of network weights.
2. Smooth Tanh layers permit first and second coordinate derivatives.
   ReLU would have zero second derivative almost everywhere and is unsuitable here.
3. Mark time coordinates as requiring gradients. `autograd.grad` with
   `create_graph=True` keeps derivative calculations differentiable with respect
   to network weights; differentiating again gives acceleration.
4. Minimize the mean square of `u'' + u` at 64 collocation points.
   The analytic `cos(t)` trajectory is used only to evaluate the learned solution.
5. Evaluate residuals at new midpoints, check initial conditions, and compare
   the conserved energy `0.5 * (u**2 + velocity**2)` against 0.5.

Physics residual evaluation still needs autograd. `model.eval()` does not turn
off gradients; wrapping the derivative calculation in `no_grad()` would break it.
The example uses float64 for the coordinate derivatives and physical comparisons.

## Try it

- Change oscillator frequency in both the residual and reference solution.
- Add damping and replace the conservation check with an energy-decay check.
- Increase the time interval and inspect the need for more collocation points or optimization.
- Replace hard initial conditions with penalty terms and compare conditioning.

## Numerical checks

Held-out residuals, trajectory error, initial conditions and energy conservation
are checked together. A small training loss alone is not the full validation.

## Reference

[Physics-informed neural networks](https://arxiv.org/abs/1711.10561)


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
