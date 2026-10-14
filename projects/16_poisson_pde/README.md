# 16 — A physics-informed two-dimensional Poisson solver


Solve a two-dimensional elliptic PDE on the unit square with a physics-informed network.

$$-\Delta u=f,\quad f(x,y)=2\pi^2\sin(\pi x)\sin(\pi y),\quad u|_{\partial\Omega}=0.$$

The manufactured solution `sin(pi*x)*sin(pi*y)` provides a reference for
checking accuracy. It is not used as a supervised target in the training loss.

## Read the source

1. Multiply the network by `16*x*(1-x)*y*(1-y)` to satisfy all four zero Dirichlet
   boundaries exactly. The factor 16 normalizes the trial factor's peak to one.
2. Compute first derivatives with respect to `(x, y)`, then differentiate each
   component again to obtain `u_xx` and `u_yy`.
3. The `grad_outputs=ones` calculations give pointwise coordinate derivatives
   because the network evaluates each point independently; it has no batch-coupled layers.
4. Train on squared residuals of `-u_xx - u_yy - f` at 100 interior points.
   Smooth activations and `create_graph=True` allow training through these derivatives.
5. Evaluate at independent random points and inspect relative residual and solution
   errors. Check all four boundaries separately, including corners.

The solver uses float64. This small manufactured problem explains a Poisson PINN;
it is not a replacement for a general mesh-based PDE solver or an error certificate.

## Try it

- Increase collocation density and compare optimization and held-out residuals.
- Change the source term and derive the corresponding reference solution.
- Replace the hard boundary factor with boundary penalties.
- Introduce a spatially varying coefficient and derive the modified divergence-form residual.

## Numerical checks

Training and held-out residuals, relative solution error and exact boundary
conditions are checked, rather than reporting only the collocation loss.

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
