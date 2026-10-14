# 12 — Graph message passing and permutation symmetry


Classify small cycle and star graphs with two rounds of learned message passing.
Node order is randomized, and the implementation uses ordinary PyTorch matrix operations.

## Read the source

1. Each graph has eight nodes, one feature per node and an `(8, 8)` adjacency matrix.
   Undirected edges appear in both directions; diagonal entries stay zero.
2. `adjacency @ hidden` sums neighbors' representations. Add the node's own
   representation before applying a shared MLP to every node.
3. A second round lets information travel through two-hop neighborhoods.
   The sum aggregator preserves degree information that mean aggregation of
   identical node features would lose on this task.
4. Mean pooling converts node representations to a graph representation,
   followed by a two-class classifier.
5. Under node relabeling, permute features and both adjacency axes together.
   Node representations should permute in the same way (equivariance), while
   pooled graph predictions should remain unchanged (invariance).

This is a small dense, sum-aggregation example inspired by graph isomorphism
networks. It is not the full GIN architecture and makes no claim of solving general
graph isomorphism. Larger sparse graphs need different storage and batching.

## Try it

- Inspect which node representations change after each round on a star.
- Replace sum aggregation with a degree-normalized mean and compare the task.
- Add a path-graph class and update the output layer.
- Replace dense multiplication with edge-list accumulation for a sparse example.

## Numerical checks

Held-out graphs must be classified correctly. Relabeling nodes must preserve
graph logits and appropriately permute intermediate node representations.

## Reference

[How Powerful are Graph Neural Networks?](https://arxiv.org/abs/1810.00826)


## Run independently

This folder contains its own `main.py`, `requirements.txt` and teaching README.
Use Python 3.10+ and the same CPU environment as the [course setup](../../README.md).
It can be copied outside the repository and needs no other lesson's source.

```sh
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python main.py
python main.py --check
```

On macOS, install the native wheel with `python -m pip install -r requirements.txt`.
Run `python main.py --help` for options. Defaults use seed 42, one CPU thread,
and a small example-specific training budget. `--epochs` changes that budget;
shorter runs can fail convergence checks. Results are printed as JSON.
