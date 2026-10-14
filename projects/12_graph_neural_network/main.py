"""12: Graph message passing and permutation symmetry. Run with --check for numerical validation."""

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


def make_data(count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(count) % 2
    adjacency = torch.zeros(count, 8, 8)
    features = 1.0 + 0.02 * torch.randn(count, 8, 1, generator=generator)
    for graph in range(count):
        if labels[graph].item() == 0:  # An eight-node cycle.
            for node in range(8):
                adjacency[graph, node, (node + 1) % 8] = 1.0
                adjacency[graph, (node + 1) % 8, node] = 1.0
        else:  # An eight-node star.
            adjacency[graph, 0, 1:] = 1.0
            adjacency[graph, 1:, 0] = 1.0
        permutation = torch.randperm(8, generator=generator)
        adjacency[graph] = adjacency[graph][permutation][:, permutation]
    return features, adjacency, labels


class GraphClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Linear(1, 16)
        self.updates = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU())
                for _ in range(2)
            ]
        )
        self.head = nn.Linear(16, 2)

    def encode(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(features)
        for update in self.updates:
            # adjacency[i, j] sends node j's message to node i; add the node's own state.
            hidden = update(hidden + adjacency @ hidden)
        return hidden

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(features, adjacency).mean(dim=1))


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

    features, adjacency, labels = make_data(160, args.seed)
    x_test, a_test, y_test = make_data(64, args.seed + 1)
    loader = DataLoader(
        TensorDataset(features, adjacency, labels),
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 2),
    )
    model = GraphClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    initial_loss = criterion(model(features, adjacency), labels).item()
    for _ in range(args.epochs):
        model.train()
        for x, a, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x, a), y)
            loss.backward()
            optimizer.step()
    model.eval()
    permutation = torch.randperm(8, generator=torch.Generator().manual_seed(args.seed + 3))
    with torch.no_grad():
        logits = model(x_test, a_test)
        final_loss = criterion(model(features, adjacency), labels).item()
        accuracy = (logits.argmax(dim=1) == y_test).float().mean().item()
        reordered_x = x_test[:, permutation]
        reordered_a = a_test[:, permutation][:, :, permutation]
        reordered_logits = model(reordered_x, reordered_a)
        original_hidden = model.encode(x_test, a_test)
        reordered_hidden = model.encode(reordered_x, reordered_a)
    metrics = {
        "initial_loss": initial_loss,
        "train_loss": final_loss,
        "test_accuracy": accuracy,
        "node_shape": list(features.shape),
        "adjacency_shape": list(adjacency.shape),
        "permutation_max_difference": (logits - reordered_logits).abs().max().item(),
    }

    if args.check:
        assert final_loss < initial_loss * 0.1 and accuracy >= 0.95
        torch.testing.assert_close(logits, reordered_logits, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(
            original_hidden[:, permutation], reordered_hidden, rtol=1e-5, atol=1e-5
        )
    print(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
