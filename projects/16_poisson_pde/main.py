"""16: A physics-informed two-dimensional Poisson solver. Run with --check for numerical validation."""

import argparse
import json

import torch
from torch import nn


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return result


class Solution(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1)
        )

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        x, y = coordinates[:, :1], coordinates[:, 1:]
        boundary_factor = 16.0 * x * (1.0 - x) * y * (1.0 - y)
        return boundary_factor * self.network(2.0 * coordinates - 1.0)


def exact(coordinates: torch.Tensor) -> torch.Tensor:
    return (torch.pi * coordinates[:, :1]).sin() * (torch.pi * coordinates[:, 1:]).sin()


def forcing(coordinates: torch.Tensor) -> torch.Tensor:
    return 2.0 * torch.pi**2 * exact(coordinates)


def residual(model: nn.Module, points: torch.Tensor) -> torch.Tensor:
    coordinates = points.detach().clone().requires_grad_(True)
    value = model(coordinates)
    gradient = torch.autograd.grad(value, coordinates, torch.ones_like(value), create_graph=True)[0]
    xx = torch.autograd.grad(
        gradient[:, :1], coordinates, torch.ones_like(value), create_graph=True
    )[0][:, :1]
    yy = torch.autograd.grad(
        gradient[:, 1:], coordinates, torch.ones_like(value), create_graph=True
    )[0][:, 1:]
    # -Laplacian(u) = f on the unit square; the trial function fixes all four boundaries.
    return -xx - yy - forcing(coordinates)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--epochs", type=positive_int, default=1200)
    parser.add_argument(
        "--check", action="store_true", help="validate the lesson's numerical behavior"
    )

    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(
        torch.float64
    )  # Accurate coordinate derivatives and numerical comparisons.

    axis = torch.linspace(0.0, 1.0, 12)[1:-1]
    grid_x, grid_y = torch.meshgrid(axis, axis, indexing="ij")
    collocation = torch.stack((grid_x.flatten(), grid_y.flatten()), dim=1)
    model = Solution()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    initial_loss = residual(model, collocation).square().mean().item()
    for _ in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = residual(model, collocation).square().mean()
        loss.backward()
        optimizer.step()
    model.eval()
    held_out = torch.rand(128, 2, generator=torch.Generator().manual_seed(args.seed + 1))
    held_out_residual = residual(model, held_out)
    relative_residual = (
        held_out_residual.square().mean().sqrt() / forcing(held_out).square().mean().sqrt()
    ).item()
    edge = torch.linspace(0.0, 1.0, 25).unsqueeze(1)
    boundaries = torch.cat(
        (
            torch.cat((torch.zeros_like(edge), edge), dim=1),
            torch.cat((torch.ones_like(edge), edge), dim=1),
            torch.cat((edge, torch.zeros_like(edge)), dim=1),
            torch.cat((edge, torch.ones_like(edge)), dim=1),
        )
    )
    with torch.no_grad():
        reference = exact(held_out)  # Reference labels are used only for evaluation.
        predicted = model(held_out)
        relative_error = (
            (predicted - reference).square().mean().sqrt() / reference.square().mean().sqrt()
        ).item()
        max_error = (predicted - reference).abs().max().item()
        boundary_error = model(boundaries).abs().max().item()
    metrics = {
        "initial_residual_mse": initial_loss,
        "training_residual_mse": loss.item(),
        "held_out_relative_residual": relative_residual,
        "relative_solution_error": relative_error,
        "max_solution_error": max_error,
        "max_boundary_error": boundary_error,
        "collocation_points": len(collocation),
        "dimension": 2,
    }

    if args.check:
        assert loss.item() < initial_loss * 0.01
        assert relative_residual < 0.03, "held-out Poisson residual is too large"
        assert relative_error < 0.02 and max_error < 0.03, "manufactured-solution comparison failed"
        assert boundary_error < 1e-12, "Dirichlet boundary conditions were not preserved"
    print(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
