"""09: Embeddings, attention and padding masks. Run with --check to validate the result."""

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


# A deliberately small synthetic vocabulary; labels depend on one signal word.
VOCABULARY = {0: "PAD", 1: "the", 2: "movie", 3: "was", 4: "quite", 5: "today", 6: "good", 7: "bad"}


def make_data(count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(count) % 2
    tokens = torch.randint(1, 6, (count, 8), generator=generator)
    lengths = torch.randint(4, 9, (count,), generator=generator)
    for i in range(count):
        position = torch.randint(0, int(lengths[i]), (1,), generator=generator).item()
        tokens[i, position] = 6 if labels[i].item() == 0 else 7
        tokens[i, lengths[i] :] = 0
    return tokens, labels


class Classifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.words = nn.Embedding(len(VOCABULARY), 16, padding_idx=0)
        self.positions = nn.Embedding(8, 16)
        layer = nn.TransformerEncoderLayer(
            d_model=16, nhead=2, dim_feedforward=32, dropout=0.0, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.head = nn.Linear(16, 2)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        padding = tokens == 0  # For this API, True marks keys that attention must ignore.
        positions = torch.arange(tokens.size(1), device=tokens.device)
        embedded = self.words(tokens) + self.positions(positions)
        encoded = self.encoder(embedded, src_key_padding_mask=padding)
        valid = (~padding).unsqueeze(-1)
        pooled = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        return self.head(pooled)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--check", action="store_true", help="run the lesson's numerical checks")
    parser.add_argument("--epochs", type=positive_int, default=20)

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
    model = Classifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    initial_loss = criterion(model(x_train), y_train).item()
    for _ in range(args.epochs):
        model.train()
        for tokens, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(tokens), labels)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(x_test)
        accuracy = (logits.argmax(dim=1) == y_test).float().mean().item()
        final_loss = criterion(model(x_train), y_train).item()
        # Masked padding must not influence valid tokens or the pooled prediction.
        model.words.weight[0].fill_(100.0)
        changed_padding_logits = model(x_test)
    metrics = {
        "token_shape": list(x_train.shape),
        "vocabulary_size": len(VOCABULARY),
        "initial_loss": initial_loss,
        "train_loss": final_loss,
        "test_accuracy": accuracy,
        "padding_max_difference": (logits - changed_padding_logits).abs().max().item(),
    }

    if args.check:
        assert final_loss < initial_loss * 0.1
        assert accuracy >= 0.95, "attention classifier failed on independent token sequences"
        assert logits.shape == (96, 2) and x_train.dtype == torch.long
        torch.testing.assert_close(logits, changed_padding_logits, rtol=1e-5, atol=1e-5)
    print(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
