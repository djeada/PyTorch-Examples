"""17: Differentiable heat flow and inverse parameter fitting. Run with --check for numerical validation."""

import argparse
import json

import torch
from torch import nn


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return result


MAX_DIFFUSIVITY = 0.3


def simulate(
    diffusivity: torch.Tensor, points: int = 21, steps: int = 100, duration: float = 0.1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.linspace(0.0, 1.0, points)
    dx = 1.0 / (points - 1)
    dt = duration / steps
    if MAX_DIFFUSIVITY * dt / dx**2 > 0.5:
        raise ValueError("The maximum allowed diffusivity violates explicit heat-step stability.")
    state = torch.cat((torch.zeros(1), torch.sin(torch.pi * x[1:-1]), torch.zeros(1)))
    ratio = diffusivity * dt / dx**2
    trajectory = [state]
    for _ in range(steps):
        laplacian = state[2:] - 2.0 * state[1:-1] + state[:-2]
        interior = state[1:-1] + ratio * laplacian
        state = torch.cat((torch.zeros(1), interior, torch.zeros(1)))
        trajectory.append(state)
    return torch.stack(trajectory), x, torch.linspace(0.0, duration, steps + 1)


def exact(diffusivity: float, x: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    return torch.exp(-diffusivity * torch.pi**2 * times[:, None]) * torch.sin(torch.pi * x[None, :])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--epochs", type=positive_int, default=180)
    parser.add_argument(
        "--check", action="store_true", help="validate the lesson's numerical behavior"
    )

    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(
        torch.float64
    )  # Accurate coordinate derivatives and numerical comparisons.

    true_diffusivity = 0.15
    x = torch.linspace(0.0, 1.0, 21)
    observed_steps = torch.tensor([25, 50, 75, 100])
    sensors = torch.tensor([4, 8, 12, 16])
    observed_times = observed_steps * 0.001
    # Generate observations from the continuous analytic solution, not the training discretization.
    observations = exact(true_diffusivity, x[sensors], observed_times)
    raw = nn.Parameter(-0.8 + 0.2 * torch.randn(()))
    optimizer = torch.optim.Adam([raw], lr=0.1)
    with torch.no_grad():
        initial_trajectory, _, _ = simulate(MAX_DIFFUSIVITY * raw.sigmoid())
        initial_loss = (
            (initial_trajectory[observed_steps][:, sensors] - observations).square().mean().item()
        )
    first_gradient = None
    for _ in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        diffusivity = (
            MAX_DIFFUSIVITY * raw.sigmoid()
        )  # Positive and bounded for stable explicit stepping.
        trajectory, _, _ = simulate(diffusivity)
        loss = (trajectory[observed_steps][:, sensors] - observations).square().mean()
        loss.backward()
        if first_gradient is None:
            first_gradient = raw.grad.item()
        optimizer.step()
    with torch.no_grad():
        estimate = MAX_DIFFUSIVITY * raw.sigmoid()
        trajectory, x, times = simulate(estimate)
        reference = exact(true_diffusivity, x, times)
        max_error = (trajectory - reference).abs().max().item()
        energy = trajectory.square().sum(dim=1) / 20.0
        largest_energy_increase = (energy[1:] - energy[:-1]).max().item()
        refined, _, _ = simulate(estimate, points=41, steps=400)
        refinement_error = (trajectory - refined[::4, ::2]).abs().max().item()
        boundary_error = trajectory[:, [0, -1]].abs().max().item()
        ratio = estimate.item() * 0.001 / 0.05**2
    metrics = {
        "initial_observation_mse": initial_loss,
        "observation_mse": loss.item(),
        "true_diffusivity": true_diffusivity,
        "estimated_diffusivity": estimate.item(),
        "first_parameter_gradient": first_gradient,
        "stability_ratio": ratio,
        "max_solution_error": max_error,
        "max_boundary_error": boundary_error,
        "largest_energy_increase": largest_energy_increase,
        "grid_refinement_max_difference": refinement_error,
    }

    if args.check:
        assert abs(first_gradient) > 1e-8 and torch.isfinite(torch.tensor(first_gradient))
        assert loss.item() < initial_loss * 0.01 and abs(estimate.item() - true_diffusivity) < 0.01
        assert 0.0 < ratio <= 0.5 and boundary_error < 1e-12
        assert largest_energy_increase <= 1e-12, (
            "heat flow should dissipate squared-amplitude energy"
        )
        assert max_error < 0.005 and refinement_error < 0.002
    print(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
