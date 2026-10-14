"""02: Autograd and manual gradient descent. Run with --check to validate the result."""

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
    parser.add_argument("--epochs", type=positive_int, default=80)

    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    x.square().sum().backward()
    first_gradient = x.grad.clone()
    # A fresh graph, but the same leaf: backward accumulates into x.grad.
    x.square().sum().backward()
    accumulated_gradient = x.grad.clone()
    x.grad.zero_()

    # For vector outputs, supply the vector in a vector-Jacobian product.
    outputs = torch.stack((x[0] * x[1], x[1] * x[2]))
    vjp = torch.autograd.grad(outputs, x, grad_outputs=torch.tensor([1.0, 2.0]))[0]

    inputs = torch.linspace(-2.0, 2.0, 41)
    targets = 3.0 * inputs + 2.0
    weight = torch.tensor(0.0, requires_grad=True)
    bias = torch.tensor(0.0, requires_grad=True)
    initial_loss = ((weight * inputs + bias - targets) ** 2).mean().item()
    for _ in range(args.epochs):
        loss = ((weight * inputs + bias - targets) ** 2).mean()
        loss.backward()
        # Update parameters without recording the update in the next computation graph.
        with torch.no_grad():
            weight -= 0.1 * weight.grad
            bias -= 0.1 * bias.grad
        weight.grad.zero_()
        bias.grad.zero_()
    final_loss = ((weight * inputs + bias - targets) ** 2).mean().item()
    metrics = {
        "first_gradient": first_gradient.tolist(),
        "accumulated_gradient": accumulated_gradient.tolist(),
        "vjp": vjp.tolist(),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "weight": weight.item(),
        "bias": bias.item(),
    }

    if args.check:
        torch.testing.assert_close(first_gradient, torch.tensor([2.0, 4.0, 6.0]))
        torch.testing.assert_close(accumulated_gradient, torch.tensor([4.0, 8.0, 12.0]))
        torch.testing.assert_close(vjp, torch.tensor([2.0, 7.0, 4.0]))
        assert final_loss < 1e-8, "manual gradient descent did not converge"
        assert abs(weight.item() - 3.0) < 1e-4 and abs(bias.item() - 2.0) < 1e-4
    print(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
