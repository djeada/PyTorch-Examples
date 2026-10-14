"""04: Datasets and mini-batches. Run with --check to validate the result."""

import argparse
import json

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


class BlobDataset(Dataset):
    def __init__(self, samples_per_class: int, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        centers = torch.tensor([[-2.0, -1.0], [2.0, -1.0], [0.0, 2.0]])
        self.labels = torch.arange(3).repeat_interleave(samples_per_class)
        self.features = centers[self.labels] + 0.35 * torch.randn(
            len(self.labels), 2, generator=generator
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--check", action="store_true", help="run the lesson's numerical checks")
    parser.add_argument("--epochs", type=positive_int, default=30)

    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    train = BlobDataset(50, args.seed)
    test = BlobDataset(30, args.seed + 1)
    loader = DataLoader(
        train,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    model = nn.Linear(2, 3)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with torch.no_grad():
        initial_loss = criterion(model(train.features), train.labels).item()
    for _ in range(args.epochs):
        model.train()
        total_loss = 0.0
        samples_seen = 0
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
            # Weight by batch size: the last batch is smaller than 32.
            total_loss += loss.item() * len(labels)
            samples_seen += len(labels)
        epoch_loss = total_loss / samples_seen
    model.eval()
    with torch.no_grad():
        accuracy = (model(test.features).argmax(dim=1) == test.labels).float().mean().item()
    metrics = {
        "initial_loss": initial_loss,
        "epoch_loss": epoch_loss,
        "test_accuracy": accuracy,
        "samples_seen": samples_seen,
        "batch_count": len(loader),
        "last_batch_size": len(train) % loader.batch_size,
    }

    if args.check:
        assert samples_seen == len(train), "the final partial batch was lost"
        assert epoch_loss < initial_loss * 0.1
        assert accuracy >= 0.95, "held-out blob classification failed"
        assert train.features.shape == (150, 2) and train.labels.dtype == torch.long
    print(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
