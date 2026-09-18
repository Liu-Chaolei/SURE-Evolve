"""Faithful projection of native ASR ideas onto the fixed Zipformer interface."""
KEYS = ("num-encoder-layers", "encoder-dim", "feedforward-dim", "encoder-unmasked-dim")


def architecture_arguments(raw):
    if not isinstance(raw, dict) or set(raw) != set(KEYS):
        raise ValueError("ASR projection requires exactly four architecture arguments")
    values = {}
    for key in KEYS:
        if not isinstance(raw[key], str):
            raise ValueError("Architecture arguments must be comma-separated strings")
        row = [int(x.strip()) for x in raw[key].split(",")]
        if len(row) != 6 or min(row) < (0 if key == "encoder-unmasked-dim" else 1):
            raise ValueError("Architecture requires six valid stages")
        if key != "num-encoder-layers" and any(x % 32 for x in row):
            raise ValueError("Architecture dimensions must be multiples of 32")
        values[key] = row
    if any(a > b for a, b in zip(values["encoder-unmasked-dim"], values["encoder-dim"])):
        raise ValueError("Unmasked dimensions exceed encoder dimensions")
    if sum(values["num-encoder-layers"]) > 20 or max(values["encoder-dim"]) > 512 or max(values["feedforward-dim"]) > 1536:
        raise ValueError("Architecture exceeds the fixed ASR search envelope")
    return {key: ",".join(map(str, values[key])) for key in KEYS}


REVIEW_PROMPT = """
This is fixed-budget ASR Zipformer architecture search. When accepting, also return
architecture_arguments containing exactly num-encoder-layers, encoder-dim,
feedforward-dim, encoder-unmasked-dim, each a six-stage comma-separated integer string.
Project only the supplied native fused mechanism, never invent an unrelated replacement.
Dimensions must be multiples of 32; unmasked dimensions cannot exceed encoder dimensions;
sum(layers)<=20, max(encoder-dim)<=512, max(feedforward-dim)<=1536.
If the fused mechanism cannot be faithfully implemented through these four arguments,
reject it. Keep training, data, loss, optimizer, precision, duration and decoding fixed.
Do not repeat the baseline, accepted candidates or previously measured architectures.
The ablation list must be empty. No auxiliary experiment may be scheduled.
"""
