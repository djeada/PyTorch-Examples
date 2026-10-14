"""08: Sequence forecasting with a GRU. Run with --check to validate the result."""

import argparse
import json

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def make_data(count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    phase = 2.0 * torch.pi * torch.rand(count, 1, generator=generator)
    frequency = 0.15 + 0.2 * torch.rand(count, 1, generator=generator)
    time = torch.arange(13).unsqueeze(0)
    waves = torch.sin(phase + frequency * time)
    waves += 0.02 * torch.randn(count, 13, generator=generator)
    # Each example is an independently generated wave: no overlapping split windows.
    return waves[:, :12].unsqueeze(-1), waves[:, 12:13]


class Forecaster(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.recurrent = nn.GRU(input_size=1, hidden_size=24, batch_first=True)
        self.head = nn.Linear(24, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.recurrent(x)
        return self.head(sequence[:, -1, :])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--check", action="store_true", help="run the lesson's numerical checks")
    parser.add_argument("--epochs", type=positive_int, default=40)

    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    x_train, y_train = make_data(256, args.seed)
    x_test, y_test = make_data(96, args.seed + 1)
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    model = Forecaster()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    initial_loss = criterion(model(x_train), y_train).item()
    for _ in range(args.epochs):
        model.train()
        for sequence, target in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(sequence), target)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        prediction = model(x_test)
        test_loss = criterion(prediction, y_test).item()
        persistence_loss = criterion(x_test[:, -1, :], y_test).item()
        train_loss = criterion(model(x_train), y_train).item()
    metrics = {
        "input_shape": list(x_train.shape),
        "target_shape": list(y_train.shape),
        "initial_loss": initial_loss,
        "train_mse": train_loss,
        "test_mse": test_loss,
        "last_value_baseline_mse": persistence_loss,
    }

    if args.check:
        assert x_train.shape == (256, 12, 1) and prediction.shape == y_test.shape == (96, 1)
        assert train_loss < initial_loss * 0.1
        assert test_loss < 0.025 and test_loss < persistence_loss * 0.5, (
            "forecast did not beat the last-value baseline"
        )
    print(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
