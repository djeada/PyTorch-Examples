"""05: Nonlinear classification with an MLP. Run with --check to validate the result."""

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
    labels = torch.arange(count) % 2
    angle = 2.0 * torch.pi * torch.rand(count, generator=generator)
    radius = torch.where(labels == 0, 0.5, 1.2) + 0.05 * torch.randn(count, generator=generator)
    features = torch.stack((radius * angle.cos(), radius * angle.sin()), dim=1)
    return features, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--check", action="store_true", help="run the lesson's numerical checks")
    parser.add_argument("--epochs", type=positive_int, default=25)

    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    x_train, y_train = make_data(256, args.seed)
    x_test, y_test = make_data(128, args.seed + 1)
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    model = nn.Sequential(
        nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2)
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    initial_loss = criterion(model(x_train), y_train).item()
    for _ in range(args.epochs):
        model.train()
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        final_loss = criterion(model(x_train), y_train).item()
        accuracy = (model(x_test).argmax(dim=1) == y_test).float().mean().item()
    metrics = {
        "initial_loss": initial_loss,
        "train_loss": final_loss,
        "test_accuracy": accuracy,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }

    if args.check:
        assert final_loss < initial_loss * 0.1
        assert accuracy >= 0.95, "the network did not learn the nonlinear ring boundary"
        assert model(x_test).shape == (128, 2)
    print(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
