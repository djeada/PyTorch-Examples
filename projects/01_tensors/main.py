"""01: Tensors, shapes and broadcasting. Run with --check to validate the result."""

import argparse
import json

import torch


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--check", action="store_true", help="run the lesson's numerical checks")

    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    matrix = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    broadcast = matrix + torch.tensor([10.0, 20.0, 30.0])
    product = matrix @ matrix.T  # (2, 3) @ (3, 2) -> (2, 2)

    # A view of this contiguous tensor shares its storage; a clone owns new storage.
    original = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    view = original.view(3, 2)
    copy = original.clone()
    view[0, 0] = -1.0
    metrics = {
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "device": str(matrix.device),
        "broadcast": broadcast.tolist(),
        "matrix_product": product.tolist(),
        "shared_first_value": original[0, 0].item(),
        "copied_first_value": copy[0, 0].item(),
    }

    if args.check:
        torch.testing.assert_close(
            broadcast, torch.tensor([[11.0, 22.0, 33.0], [14.0, 25.0, 36.0]])
        )
        torch.testing.assert_close(product, torch.tensor([[14.0, 32.0], [32.0, 77.0]]))
        assert original[0, 0].item() == -1.0 and copy[0, 0].item() == 0.0
        assert matrix.dtype == torch.float32 and matrix.device.type == "cpu"
    print(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
