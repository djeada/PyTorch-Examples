# 10 — Saving, loading, inference and resuming training


Train a regressor, save its weights and optimizer state, reload it for inference,
then verify that resumed training produces the same next Adam update.

## Read the source

1. Save state dictionaries, completed epoch count, seed and CPU RNG state.
   The model architecture remains ordinary Python code in this folder.
2. Load the checkpoint using `map_location="cpu"` and `weights_only=True`.
   Instantiate the same architecture before calling `load_state_dict`.
3. Instantiate an optimizer over the restored model's parameters, then restore
   its state dictionary. Adam's running moments and step count affect its next update.
4. Restore RNG state, switch the model to `eval()`, and run predictions under
   `inference_mode()` when gradients will not be needed.
5. Compare reloaded predictions, then advance both the original and restored
   training states by one deterministic full-batch step and compare their weights.

This resume check covers this CPU, full-batch example. A stochastic production
pipeline may also need sampler progress, worker RNG states, a scheduler and other state.

## Checkpoint output

```sh
python main.py --output outputs/my_model.pt
```

Ordinary runs default to `outputs/checkpoint.pt`. `--check` uses a temporary file
and removes it afterward unless `--output` explicitly requests a destination.

## Try it

- Restore weights but omit optimizer state and compare the next Adam update.
- Change the architecture and inspect the state-dictionary mismatch error.
- Save after a different number of epochs and compare held-out predictions.
- Add a learning-rate scheduler and include its state in the checkpoint.

## Numerical checks

The checkpoint must exist when loaded, predictions must survive the round trip
exactly, and restoring optimizer state must reproduce the next parameter update.


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
