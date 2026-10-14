# 09 — Embeddings, attention and padding masks


Classify short generated token sequences containing either a positive or negative
signal word. Variable-length sequences are padded to eight tokens.

## Read the source

1. Token IDs are integers. An embedding maps each ID to 16 trainable features;
   learned positional embeddings provide information about sequence position.
2. The encoder uses two attention heads. Attention combines information across
   the full input sequence; this classification task needs no causal mask.
3. For `src_key_padding_mask`, **True means ignore this key**. Supply one boolean
   per token so padding cannot contribute to the representation of valid tokens.
4. Also exclude padding from mean pooling. Masking attention alone does not make
   an unmasked average over padded outputs correct.
5. Classify the pooled representation with two logits and evaluate on new sequences.
   This small synthetic task explains the APIs; it is not a pretrained language model.

## Try it

- Remove the positional embeddings and consider which tasks would lose order information.
- Create a label rule that depends on word order rather than one keyword.
- Change the number of heads while keeping the embedding width divisible by it.
- Remove the padding mask or masked pooling and compare predictions for padded sequences.

## Numerical checks

Held-out accuracy must reach 95%, output shapes and token dtype must match,
and changing the padding embedding must not change valid sequence predictions.


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
