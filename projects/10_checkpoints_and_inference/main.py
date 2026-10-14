"""10: Saving, loading, inference and resuming training. Run with --check to validate the result."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

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


def train_step(
    model: nn.Module, optimizer: torch.optim.Optimizer, x: torch.Tensor, y: torch.Tensor
) -> None:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.mse_loss(model(x), y)
    loss.backward()
    optimizer.step()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--check", action="store_true", help="run the lesson's numerical checks")
    parser.add_argument("--epochs", type=positive_int, default=200)
    parser.add_argument(
        "--output", type=Path, help="checkpoint destination; default: outputs/checkpoint.pt"
    )
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    x_train, y_train = make_data(128, args.seed)
    x_test, y_test = make_data(64, args.seed + 1)
    model = nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    initial_loss = nn.functional.mse_loss(model(x_train), y_train).item()
    for _ in range(args.epochs):
        train_step(model, optimizer, x_train, y_train)
    model.eval()
    with torch.inference_mode():
        expected_prediction = model(x_test)

    # --check uses a temporary file unless a destination was explicitly requested.
    with TemporaryDirectory() as temporary:
        output = args.output or (
            Path(temporary) / "checkpoint.pt" if args.check else Path("outputs/checkpoint.pt")
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epochs": args.epochs,
                "seed": args.seed,
                "rng_state": torch.get_rng_state(),
            },
            output,
        )
        checkpoint = torch.load(output, map_location="cpu", weights_only=True)
        restored = nn.Linear(1, 1)
        restored.load_state_dict(checkpoint["model"])
        restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.05)
        restored_optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["rng_state"])
        restored.eval()
        with torch.inference_mode():
            restored_prediction = restored(x_test)
            test_loss = nn.functional.mse_loss(restored_prediction, y_test).item()
        inference_equal = torch.equal(expected_prediction, restored_prediction)

        # On this deterministic full-batch problem, restored Adam state must yield the same next step.
        train_step(model, optimizer, x_train, y_train)
        train_step(restored, restored_optimizer, x_train, y_train)
        resume_equal = all(
            torch.equal(a, b)
            for a, b in zip(model.parameters(), restored.parameters(), strict=True)
        )
        checkpoint_exists = output.is_file()
        metrics = {
            "checkpoint": "temporary (--check)"
            if args.check and args.output is None
            else str(output),
            "initial_loss": initial_loss,
            "test_mse": test_loss,
            "saved_epochs": checkpoint["epochs"],
            "inference_equal": inference_equal,
            "resumed_update_equal": resume_equal,
        }

    if args.check:
        assert checkpoint_exists and checkpoint["epochs"] == args.epochs
        assert inference_equal, "saving/loading changed the predictions"
        assert resume_equal, "restoring the optimizer did not reproduce the next Adam update"
        assert test_loss < 0.03, "checkpointed model failed on held-out regression samples"
    print(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
