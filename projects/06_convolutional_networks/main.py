"""06: Image classification with a CNN. Run with --check to validate the result."""

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
    rows, columns = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
    labels = torch.arange(count) % 3
    centers = torch.randint(4, 12, (count,), generator=generator)
    images = torch.empty(count, 1, 16, 16)
    for i in range(count):
        if labels[i].item() == 0:
            pattern = (rows - centers[i]).abs() <= 1
        elif labels[i].item() == 1:
            pattern = (columns - centers[i]).abs() <= 1
        else:
            pattern = (rows - columns - (centers[i] - 8)).abs() <= 1
        noise = 0.08 * torch.randn(16, 16, generator=generator)
        images[i, 0] = (pattern.float() + noise).clamp(0.0, 1.0)
    return images, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--check", action="store_true", help="run the lesson's numerical checks")
    parser.add_argument("--epochs", type=positive_int, default=25)

    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    x_train, y_train = make_data(240, args.seed)
    x_test, y_test = make_data(90, args.seed + 1)
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    model = nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(8, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 3),
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    initial_loss = criterion(model(x_train), y_train).item()
    for _ in range(args.epochs):
        model.train()
        for images, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        final_loss = criterion(model(x_train), y_train).item()
        accuracy = (model(x_test).argmax(dim=1) == y_test).float().mean().item()
    metrics = {
        "image_shape": list(x_train.shape),
        "initial_loss": initial_loss,
        "train_loss": final_loss,
        "test_accuracy": accuracy,
        "classes": 3,
    }

    if args.check:
        assert x_train.shape == (240, 1, 16, 16)
        assert final_loss < initial_loss * 0.2
        assert accuracy >= 0.90, "the CNN did not recognize held-out bar orientations"
    print(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
