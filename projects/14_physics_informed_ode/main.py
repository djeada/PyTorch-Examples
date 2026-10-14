"""14: A physics-informed harmonic oscillator. Run with --check for numerical validation."""

import argparse
import json

import torch
from torch import nn


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return result


class Oscillator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 24), nn.Tanh(), nn.Linear(24, 24), nn.Tanh(), nn.Linear(24, 1)
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        # This trial function satisfies u(0)=1 and u'(0)=0 for every set of weights.
        return 1.0 + time.square() * self.network(time)


def physics(
    model: nn.Module, coordinates: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    time = coordinates.detach().clone().requires_grad_(True)
    displacement = model(time)
    velocity = torch.autograd.grad(
        displacement, time, torch.ones_like(displacement), create_graph=True
    )[0]
    acceleration = torch.autograd.grad(
        velocity, time, torch.ones_like(velocity), create_graph=True
    )[0]
    return acceleration + displacement, displacement, velocity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--epochs", type=positive_int, default=800)
    parser.add_argument(
        "--check", action="store_true", help="validate the lesson's numerical behavior"
    )

    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(
        torch.float64
    )  # Accurate coordinate derivatives and numerical comparisons.

    collocation = torch.linspace(0.0, 2.0, 64).unsqueeze(1)
    model = Oscillator()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    initial_loss = physics(model, collocation)[0].square().mean().item()
    for _ in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        residual, _, _ = physics(model, collocation)
        loss = residual.square().mean()  # No trajectory labels are used for training.
        loss.backward()
        optimizer.step()
    model.eval()
    held_out = (collocation[:-1] + collocation[1:]) / 2.0
    residual, displacement, velocity = physics(model, held_out)
    test_residual = residual.square().mean().item()
    max_error = (displacement - held_out.cos()).abs().max().item()
    energy = 0.5 * (displacement.square() + velocity.square())
    energy_error = (energy - 0.5).abs().max().item()
    _, initial_displacement, initial_velocity = physics(model, torch.zeros(1, 1))
    metrics = {
        "initial_residual_mse": initial_loss,
        "held_out_residual_mse": test_residual,
        "max_solution_error": max_error,
        "max_energy_error": energy_error,
        "initial_displacement": initial_displacement.item(),
        "initial_velocity": initial_velocity.item(),
        "dtype": str(displacement.dtype),
    }

    if args.check:
        assert test_residual < initial_loss * 0.05 and test_residual < 0.002
        assert max_error < 0.02 and energy_error < 0.02, (
            "oscillator solution violates its reference or conserved energy"
        )
        torch.testing.assert_close(initial_displacement, torch.ones(1, 1), rtol=0, atol=1e-12)
        torch.testing.assert_close(initial_velocity, torch.zeros(1, 1), rtol=0, atol=1e-12)
    print(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
