# 06 — Image classification with a CNN


Recognize horizontal, vertical and diagonal bars in generated noisy grayscale images.
Each image is 16 by 16 pixels, and the bar position varies between samples.

## Read the source

1. Images use `(batch, channels, height, width)` layout: `(240, 1, 16, 16)`.
   Class labels are a separate integer tensor.
2. `Conv2d` learns spatial kernels whose weights are reused across the image.
   Padding of one preserves spatial dimensions for a three-by-three kernel.
3. ReLU introduces nonlinearity and max pooling reduces the spatial resolution.
4. A second convolution learns features from the first layer's feature maps.
   Adaptive average pooling produces one value per channel before the classifier.
5. Use raw logits with cross entropy and evaluate on independently generated images.

## Try it

- Trace the tensor shape after every layer.
- Change kernel size and calculate the padding needed to preserve image dimensions.
- Add a new orientation and update both the generated labels and output layer.
- Increase image noise or vary bar thickness to explore what the kernels learn.

## Numerical checks

The image layout is verified, loss must decrease, and held-out accuracy must
reach 90%. This is a synthetic pattern task, not a natural-image benchmark.


## Run independently

Copy this folder anywhere. It contains its own source and dependency file;
no repository package, other lesson, dataset download or GPU is required.
Use Python 3.10+ and install the dependencies in a virtual environment.

From this folder on Linux or Windows with a CPU PyTorch installation:

```sh
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python main.py
python main.py --check
```

On macOS, install with `python -m pip install -r requirements.txt` using the
native PyTorch wheel. See the [course setup](../../README.md) for virtual
environment commands. The folder remains runnable if copied without that README.

Run `python main.py --help` for options. The default seed is 42 and the default
CPU thread count is one. Training lessons expose `--epochs`; the numerical
checks target their default training budgets. Every run reports results as JSON.
