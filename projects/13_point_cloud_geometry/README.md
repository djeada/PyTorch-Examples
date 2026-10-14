# 13 — Learning from unordered point clouds


Recognize noisy circles and lines from unordered two-dimensional point sets.
Each cloud is randomly rotated and translated before entering the classifier.

## Read the source

1. Inputs have shape `(batch, points, coordinates)`, or `(256, 32, 2)` here.
   A point cloud has no meaningful storage order.
2. Subtract each cloud's centroid to remove its translation. This preprocessing
   makes predictions insensitive to adding the same offset to all points.
3. A shared MLP transforms each point independently. Max pooling over points
   provides a symmetric, order-independent representation.
4. A second MLP classifies the pooled representation. This follows the central
   shared-feature/symmetric-pooling idea of PointNet, without its transform networks.
5. Rotation augmentation exposes the model to different orientations, but the
   architecture does not guarantee rotation invariance. Permutation invariance
   and translation removal have stronger structural guarantees here.

## Try it

- Increase or decrease the number of sampled points.
- Replace max pooling with mean pooling and compare behavior with outliers.
- Add a square class or extend coordinates to three dimensions.
- Inspect predictions after a rotation and compare learned robustness with
  the guaranteed behavior under point permutations.

## Numerical checks

Independent point clouds must be classified correctly. Shuffling point order
and translating a whole cloud must preserve predictions within floating-point tolerance.

## Reference

[PointNet](https://arxiv.org/abs/1612.00593)


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
