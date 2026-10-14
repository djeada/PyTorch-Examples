# 11 — A tiny causal language model


Train a character-level, decoder-only Transformer from scratch and generate text
one token at a time. The small inline grammar produces sentences such as
"the red bird sings." No pretrained weights, tokenizer package or corpus download is used.

## Read the source

1. A fixed character vocabulary encodes strings as integer IDs. Training and
   validation text are generated independently from the same finite grammar;
   sentence combinations can overlap, so this is not a test of novel language understanding.
2. Each input window has 32 tokens. Targets are shifted by one character, so
   each output predicts the next token rather than a sentence class.
3. Token and position embeddings feed two residual Transformer blocks with
   pre-normalization, four attention heads, and a feed-forward network.
4. `scaled_dot_product_attention(..., is_causal=True)` prevents a token from
   attending to later tokens. This differs from the bidirectional encoder in project 09.
5. Cross entropy acts on raw next-token logits. Perplexity is `exp(mean loss)`
   with the natural-log cross entropy used here.
6. Generation repeatedly samples the last position's distribution at temperature
   0.8 and appends one token. Context is cropped to the most recent 32 characters.

## Generate text

```sh
python main.py --prompt "a blue " --tokens 80
```

This miniature model teaches LLM mechanics; its vocabulary and training text
are deliberately small. It is not an instruction-following or pretrained large model.
`--steps` controls random-window optimizer updates; they are not full-corpus epochs.

## Try it

- Change the grammar or add an inline text corpus, updating the fixed vocabulary.
- Adjust embedding width, number of heads and context length together.
- Compare greedy decoding with sampling or change temperature in `generate`.
- Remove the causal restriction and inspect why next-token training can cheat.

## Numerical checks

Validation loss and perplexity must improve. Changing future tokens must leave
earlier logits unchanged, and generation must produce the requested token count.

## Reference

[PyTorch scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention)


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
and a small example-specific training budget. `--steps` changes that budget;
shorter runs can fail convergence checks. Results are printed as JSON.
