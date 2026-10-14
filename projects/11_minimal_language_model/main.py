"""11: A tiny causal language model. Run with --check for numerical validation."""

import argparse
import json
import math

import torch
from torch import nn
from torch.nn import functional as F


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return result


CONTEXT = 32
WORDS = (
    ("a", "the"),
    ("red", "blue", "green"),
    ("cat", "fox", "bird"),
    ("runs", "sleeps", "sings"),
)
ALPHABET = sorted(set(" ".join(word for group in WORDS for word in group) + ".\n"))
TO_ID = {character: index for index, character in enumerate(ALPHABET)}


def make_text(lines: int, seed: int) -> str:
    generator = torch.Generator().manual_seed(seed)
    sentences = []
    for _ in range(lines):
        words = [
            group[torch.randint(len(group), (1,), generator=generator).item()] for group in WORDS
        ]
        sentences.append(" ".join(words) + ".\n")
    return "".join(sentences)


def encode(text: str) -> torch.Tensor:
    return torch.tensor([TO_ID[character] for character in text], dtype=torch.long)


def batch(
    sequence: torch.Tensor, count: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(len(sequence) - CONTEXT, (count, 1), generator=generator)
    positions = starts + torch.arange(CONTEXT)
    return sequence[positions], sequence[positions + 1]


class CausalAttention(nn.Module):
    def __init__(self, width: int = 32, heads: int = 4) -> None:
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        count, length, width = x.shape
        q, k, v = (
            self.qkv(x)
            .reshape(count, length, 3, self.heads, width // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        return self.output(attended.transpose(1, 2).reshape(count, length, width))


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(32)
        self.attention = CausalAttention()
        self.norm2 = nn.LayerNorm(32)
        self.feed_forward = nn.Sequential(nn.Linear(32, 128), nn.GELU(), nn.Linear(128, 32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.feed_forward(self.norm2(x))


class LanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tokens = nn.Embedding(len(ALPHABET), 32)
        self.positions = nn.Embedding(CONTEXT, 32)
        self.blocks = nn.Sequential(Block(), Block())
        self.norm = nn.LayerNorm(32)
        self.head = nn.Linear(32, len(ALPHABET))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.size(1), device=tokens.device)
        hidden = self.tokens(tokens) + self.positions(positions)
        return self.head(self.norm(self.blocks(hidden)))


@torch.no_grad()
def generate(model: LanguageModel, prompt: str, count: int, seed: int) -> str:
    tokens = encode(prompt).unsqueeze(0)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(count):
        logits = model(tokens[:, -CONTEXT:])[:, -1, :]
        probabilities = (logits / 0.8).softmax(dim=-1)
        next_token = torch.multinomial(probabilities, 1, generator=generator)
        tokens = torch.cat((tokens, next_token), dim=1)
    return "".join(ALPHABET[index] for index in tokens[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--steps", type=positive_int, default=200)
    parser.add_argument(
        "--check", action="store_true", help="validate the lesson's numerical behavior"
    )
    parser.add_argument("--prompt", default="the ")
    parser.add_argument("--tokens", type=positive_int, default=64)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    if not args.prompt or not set(args.prompt).issubset(TO_ID):
        parser.error("prompt must be nonempty and use characters from the generated corpus")
    train = encode(make_text(256, args.seed))
    validation = encode(make_text(64, args.seed + 1))
    train_generator = torch.Generator().manual_seed(args.seed + 2)
    x_val, y_val = batch(validation, 64, torch.Generator().manual_seed(args.seed + 3))
    model = LanguageModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.01)
    with torch.no_grad():
        initial_loss = F.cross_entropy(
            model(x_val).reshape(-1, len(ALPHABET)), y_val.reshape(-1)
        ).item()
    for _ in range(args.steps):
        model.train()
        x, target = batch(train, 32, train_generator)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, len(ALPHABET)), target.reshape(-1))
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        reference = model(x_val)
        final_loss = F.cross_entropy(reference.reshape(-1, len(ALPHABET)), y_val.reshape(-1)).item()
        changed = x_val.clone()
        changed[:, 16:] = (changed[:, 16:] + 1) % len(ALPHABET)
        changed_logits = model(changed)
        causal_difference = (reference[:, :16] - changed_logits[:, :16]).abs().max().item()
    metrics = {
        "initial_validation_loss": initial_loss,
        "validation_loss": final_loss,
        "validation_perplexity": math.exp(final_loss),
        "causal_max_difference": causal_difference,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "context_length": CONTEXT,
        "generated_text": generate(model, args.prompt, args.tokens, args.seed + 4),
    }

    if args.check:
        assert final_loss < initial_loss * 0.5 and math.exp(final_loss) < 5.0
        assert reference.shape == (64, CONTEXT, len(ALPHABET))
        torch.testing.assert_close(reference[:, :16], changed_logits[:, :16], rtol=1e-5, atol=1e-5)
        assert len(metrics["generated_text"]) == len(args.prompt) + args.tokens
    print(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
