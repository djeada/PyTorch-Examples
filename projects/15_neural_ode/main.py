"""15: Learning a vector field through an ODE solver. Run with --check for numerical validation."""

import argparse
import json

import torch
from torch import nn


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return result


class VectorField(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 1))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


def integrate(field: nn.Module, initial: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    state = initial
    trajectory = [state]
    for start, end in zip(times[:-1], times[1:], strict=True):
        step = end - start
        # Classical fourth-order Runge-Kutta, written as differentiable tensor operations.
        k1 = field(state)
        k2 = field(state + 0.5 * step * k1)
        k3 = field(state + 0.5 * step * k2)
        k4 = field(state + step * k3)
        state = state + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        trajectory.append(state)
    return torch.stack(trajectory)


def reference(initial: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    return initial.unsqueeze(0) * torch.exp(-0.7 * times[:, None, None])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--epochs", type=positive_int, default=400)
    parser.add_argument(
        "--check", action="store_true", help="validate the lesson's numerical behavior"
    )

    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(
        torch.float64
    )  # Accurate coordinate derivatives and numerical comparisons.

    times = torch.linspace(0.0, 1.0, 11)
    initial = torch.tensor([[-1.5], [-0.5], [0.5], [1.5]])
    target = reference(initial, times)
    model = VectorField()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    with torch.no_grad():
        initial_loss = (integrate(model, initial, times) - target).square().mean().item()
    first_gradient_norm = None
    for _ in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = (integrate(model, initial, times) - target).square().mean()
        loss.backward()  # Gradients pass through every RK4 stage and time step.
        if first_gradient_norm is None:
            first_gradient_norm = (
                sum(p.grad.square().sum() for p in model.parameters()).sqrt().item()
            )
        optimizer.step()
    model.eval()
    new_initial = torch.tensor([[-1.0], [1.0]])
    new_times = torch.linspace(0.0, 1.2, 13)
    with torch.no_grad():
        trajectory = integrate(model, new_initial, new_times)
        test_mse = (trajectory - reference(new_initial, new_times)).square().mean().item()
        train_mse = (integrate(model, initial, times) - target).square().mean().item()
        refined = integrate(model, new_initial, torch.linspace(0.0, 1.2, 25))[::2]
        refinement_error = (trajectory - refined).abs().max().item()
        states = torch.linspace(-1.0, 1.0, 31).unsqueeze(1)
        field_error = (model(states) + 0.7 * states).abs().max().item()
    metrics = {
        "initial_loss": initial_loss,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "first_gradient_norm": first_gradient_norm,
        "max_vector_field_error": field_error,
        "step_refinement_max_difference": refinement_error,
        "trajectory_shape": list(trajectory.shape),
    }

    if args.check:
        assert first_gradient_norm > 0.0 and torch.isfinite(torch.tensor(first_gradient_norm))
        assert train_mse < initial_loss * 0.05 and test_mse < 0.001
        assert field_error < 0.06 and refinement_error < 1e-4
        assert trajectory.shape == (13, 2, 1)
    print(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
