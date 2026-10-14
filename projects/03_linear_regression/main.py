"""03: Modules, losses and optimizers. Run with --check to validate the result."""

import argparse
import json

import torch
from torch import nn


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def make_data(count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = 2.0 * torch.rand(count, 1, generator=generator) - 1.0
    y = 3.0 * x + 1.0 + 0.05 * torch.randn(count, 1, generator=generator)
    return x, y


class Regressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--check", action="store_true", help="run the lesson's numerical checks")
    parser.add_argument("--epochs", type=positive_int, default=150)

    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    x_train, y_train = make_data(128, args.seed)
    x_test, y_test = make_data(64, args.seed + 1)
    model = Regressor()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.15)
    initial_loss = criterion(model(x_train), y_train).item()
    model.train()
    for _ in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x_train), y_train)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        train_loss = criterion(model(x_train), y_train).item()
        test_loss = criterion(model(x_test), y_test).item()
    metrics = {
        "initial_loss": initial_loss,
        "train_mse": train_loss,
        "test_mse": test_loss,
        "weight": model.linear.weight.item(),
        "bias": model.linear.bias.item(),
    }

    if args.check:
        assert train_loss < initial_loss * 0.05, "training loss did not decrease enough"
        assert test_loss < 0.03, "regression failed on independently generated test samples"
        assert abs(model.linear.weight.item() - 3.0) < 0.1
        assert abs(model.linear.bias.item() - 1.0) < 0.1
    print(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
