"""07: Dropout, validation and early stopping. Run with --check to validate the result."""

import argparse
import json
from copy import deepcopy

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
    y = 1.5 * torch.sin(3.0 * x) + 0.2 * x
    y += 0.15 * torch.randn(count, 1, generator=generator)
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--check", action="store_true", help="run the lesson's numerical checks")
    parser.add_argument("--epochs", type=positive_int, default=400)

    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    x_train, y_train = make_data(64, args.seed)
    x_val, y_val = make_data(96, args.seed + 1)
    x_test, y_test = make_data(96, args.seed + 2)
    model = nn.Sequential(
        nn.Linear(1, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1)
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    model.eval()
    with torch.no_grad():
        initial_val_loss = criterion(model(x_val), y_val).item()
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()  # Dropout is active only during training.
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x_train), y_train)
        loss.backward()
        optimizer.step()
        model.eval()  # eval changes module behavior; no_grad separately disables tracking.
        with torch.no_grad():
            val_loss = criterion(model(x_val), y_val).item()
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())  # A snapshot, not live tensor references.
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= 60:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_loss = criterion(model(x_train), y_train).item()
        prediction = model(x_test)
        repeated_prediction = model(x_test)
        test_loss = criterion(prediction, y_test).item()
    metrics = {
        "initial_val_mse": initial_val_loss,
        "best_val_mse": best_val_loss,
        "test_mse": test_loss,
        "train_mse": train_loss,
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "evaluation_is_deterministic": torch.equal(prediction, repeated_prediction),
    }

    if args.check:
        assert best_val_loss < initial_val_loss * 0.2
        assert test_loss < 0.15, "the selected model failed on untouched test samples"
        assert 1 <= best_epoch <= epoch <= args.epochs
        torch.testing.assert_close(prediction, repeated_prediction, rtol=0, atol=0)
    print(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
