"""13: Learning from unordered point clouds. Run with --check for numerical validation."""

import argparse
import json

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return result


def make_data(count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(count) % 2
    angle = 2.0 * torch.pi * torch.rand(count, 32, generator=generator)
    circle = torch.stack((angle.cos(), angle.sin()), dim=-1)
    line = torch.stack(
        (2.0 * torch.rand(count, 32, generator=generator) - 1.0, torch.zeros(count, 32)), dim=-1
    )
    clouds = torch.where((labels == 0)[:, None, None], circle, line)
    clouds += 0.02 * torch.randn(count, 32, 2, generator=generator)
    rotation = 2.0 * torch.pi * torch.rand(count, generator=generator)
    matrices = torch.stack(
        (rotation.cos(), -rotation.sin(), rotation.sin(), rotation.cos()), dim=1
    ).reshape(count, 2, 2)
    clouds = clouds @ matrices.transpose(1, 2)
    clouds += 0.5 * torch.randn(count, 1, 2, generator=generator)
    return clouds, labels


class PointClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.point_features = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU()
        )
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        centered = points - points.mean(dim=1, keepdim=True)
        features = self.point_features(centered)  # One shared MLP for every point.
        pooled = features.amax(dim=1)  # A symmetric reduction over the unordered set.
        return self.head(pooled)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--epochs", type=positive_int, default=40)
    parser.add_argument(
        "--check", action="store_true", help="validate the lesson's numerical behavior"
    )

    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    x_train, y_train = make_data(256, args.seed)
    x_test, y_test = make_data(96, args.seed + 1)
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 2),
    )
    model = PointClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    initial_loss = criterion(model(x_train), y_train).item()
    for _ in range(args.epochs):
        model.train()
        for points, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(points), labels)
            loss.backward()
            optimizer.step()
    model.eval()
    permutation = torch.randperm(32, generator=torch.Generator().manual_seed(args.seed + 3))
    with torch.no_grad():
        logits = model(x_test)
        reordered = model(x_test[:, permutation])
        translated = model(x_test + torch.tensor([0.3, -0.5]))
        accuracy = (logits.argmax(dim=1) == y_test).float().mean().item()
        final_loss = criterion(model(x_train), y_train).item()
    metrics = {
        "initial_loss": initial_loss,
        "train_loss": final_loss,
        "test_accuracy": accuracy,
        "point_cloud_shape": list(x_train.shape),
        "permutation_max_difference": (logits - reordered).abs().max().item(),
        "translation_max_difference": (logits - translated).abs().max().item(),
    }

    if args.check:
        assert final_loss < initial_loss * 0.1 and accuracy >= 0.95
        torch.testing.assert_close(logits, reordered, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(logits, translated, rtol=1e-5, atol=1e-5)
    print(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
