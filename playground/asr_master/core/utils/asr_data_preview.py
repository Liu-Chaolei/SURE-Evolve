"""ASR-specific data preview generator for icefall recipes.

Reads lhotse cuts manifests, lang_bpe directories, and RESULTS.md
to produce a textual preview suitable for LLM agents.
"""

import os
import gzip
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _preview_cuts(data_dir: str, max_samples: int = 2000) -> str:
    """Generate a textual preview of lhotse cuts manifests.

    Scans data_dir/fbank/ for .cuts.jsonl.gz files and summarizes
    a bounded sample of cuts, duration stats, and speaker distribution.

    Args:
        data_dir: Path to the icefall data directory (e.g., egs/librispeech/ASR/data).

    Returns:
        A formatted string summarizing the cuts.
    """
    fbank_dir = os.path.join(data_dir, "fbank")
    if not os.path.isdir(fbank_dir):
        return f"  fbank directory not found at {fbank_dir}"

    cuts_files = sorted(Path(fbank_dir).glob("*cuts*.jsonl.gz"))
    if not cuts_files:
        return f"  No cuts manifest files found in {fbank_dir}"

    out = []
    for cuts_file in cuts_files:
        name = cuts_file.name
        num_cuts = 0
        total_duration = 0.0
        speakers = set()

        try:
            with gzip.open(cuts_file, "rt") as f:
                for line in f:
                    if num_cuts >= max_samples:
                        break
                    try:
                        cut = json.loads(line.strip())
                        num_cuts += 1
                        total_duration += cut.get("duration", 0.0)
                        # Try to extract speaker from supervisions
                        for sup in cut.get("supervisions", []):
                            spk = sup.get("speaker", None)
                            if spk:
                                speakers.add(spk)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            out.append(f"  {name}: error reading ({e})")
            continue

        hours = total_duration / 3600.0
        spk_info = f", {len(speakers)} speakers" if speakers else ""
        sample_info = f"sampled {num_cuts} cuts" if num_cuts >= max_samples else f"{num_cuts} cuts"
        out.append(f"  {name}: {sample_info}, {hours:.1f} sampled hours{spk_info}")

    return "\n".join(out)


def _preview_lang(data_dir: str) -> str:
    """Generate a textual preview of lang_bpe directory.

    Summarizes BPE vocab size and lexicon entries.

    Args:
        data_dir: Path to the icefall data directory.

    Returns:
        A formatted string summarizing the language resources.
    """
    # Find lang_bpe* directories
    data_path = Path(data_dir)
    lang_dirs = sorted(data_path.glob("lang_bpe*"))
    if not lang_dirs:
        return "  No lang_bpe* directories found"

    out = []
    for lang_dir in lang_dirs:
        name = lang_dir.name
        info = f"  {name}:"

        # Check for tokens.txt (BPE vocab)
        tokens_file = lang_dir / "tokens.txt"
        if tokens_file.exists():
            num_tokens = sum(1 for _ in open(tokens_file))
            info += f" {num_tokens} tokens"
        else:
            info += " no tokens.txt"

        # Check for L_disambig.pt or words.txt
        words_file = lang_dir / "words.txt"
        if words_file.exists():
            num_words = sum(1 for _ in open(words_file))
            info += f", {num_words} words"

        out.append(info)

    return "\n".join(out)


def _preview_results(recipe_dir: str) -> str:
    """Read RESULTS.md from the recipe directory to show baseline WER.

    Args:
        recipe_dir: Path to the icefall recipe directory (e.g., egs/librispeech/ASR/zipformer).

    Returns:
        A formatted string with RESULTS.md content (first 2000 chars).
    """
    results_file = os.path.join(recipe_dir, "RESULTS.md")
    if not os.path.exists(results_file):
        # Try parent directory (e.g., egs/librispeech/ASR/RESULTS.md)
        parent_results = os.path.join(os.path.dirname(recipe_dir), "RESULTS.md")
        if os.path.exists(parent_results):
            results_file = parent_results
        else:
            return "  No RESULTS.md found in recipe or parent directory"

    try:
        with open(results_file) as f:
            content = f.read(2000)
        return f"  RESULTS.md (first 2000 chars):\n{content}"
    except Exception as e:
        return f"  Error reading RESULTS.md: {e}"


def _preview_recipe_files(recipe_dir: str) -> str:
    """List key files in the recipe directory.

    Args:
        recipe_dir: Path to the icefall recipe directory.

    Returns:
        A formatted string listing recipe files.
    """
    if not os.path.isdir(recipe_dir):
        return f"  Recipe directory not found: {recipe_dir}"

    key_files = ["train.py", "decode.py", "model.py", "zipformer.py", "asr_datamodule.py",
                 "optim.py", "scaling.py", "subsampling.py", "finetune.py"]
    found = []
    for f in key_files:
        full_path = os.path.join(recipe_dir, f)
        if os.path.exists(full_path):
            size = os.path.getsize(full_path)
            found.append(f"  {f} ({size:,} bytes)")

    if not found:
        return "  No key recipe files found"
    return "\n".join(found)


def generate_asr_preview(icefall_root: str, recipe_dir: str, data_dir: str) -> str:
    """Generate a complete ASR data preview for the agent.

    Produces a textual summary of the icefall recipe, data manifests,
    language resources, and baseline results.

    Args:
        icefall_root: Root path of the icefall project.
        recipe_dir: Relative path of the recipe (e.g., "egs/librispeech/ASR/zipformer").
        data_dir: Path to the data directory with cuts and lang files.

    Returns:
        A formatted string with the complete ASR data preview.
    """
    out = []

    # Resolve full recipe path
    full_recipe_dir = os.path.join(icefall_root, recipe_dir) if icefall_root else recipe_dir

    # Resolve data dir: if relative, try recipe's parent data dir
    if data_dir and not os.path.isabs(data_dir):
        # e.g., egs/librispeech/ASR/data
        full_data_dir = os.path.join(icefall_root, data_dir) if icefall_root else data_dir
    else:
        full_data_dir = data_dir

    out.append("=" * 60)
    out.append("ASR RECIPE OVERVIEW")
    out.append("=" * 60)
    out.append(f"Icefall root: {icefall_root}")
    out.append(f"Recipe: {recipe_dir}")
    out.append(f"Full recipe path: {full_recipe_dir}")
    out.append(f"Data directory: {full_data_dir}")

    # Recipe files
    out.append("\n" + "-" * 40)
    out.append("KEY RECIPE FILES")
    out.append("-" * 40)
    out.append(_preview_recipe_files(full_recipe_dir))

    # Cuts manifests
    if full_data_dir and os.path.isdir(full_data_dir):
        out.append("\n" + "-" * 40)
        out.append("FBANK CUTS MANIFESTS (use this directory for --manifest-dir)")
        out.append("-" * 40)
        out.append(f"  train.py --manifest-dir should be set to: {os.path.join(full_data_dir, 'fbank')}")
        out.append("  Wrapper scripts run from ./icefall_recipe after chdir; create ./icefall_recipe/data -> ../icefall_data")
        out.append("  and pass --manifest-dir ./data/fbank, because cut manifests embed feature paths like data/fbank/.../*.lca.")
        out.append(_preview_cuts(full_data_dir))

        # Language resources
        out.append("\n" + "-" * 40)
        out.append("LANGUAGE RESOURCES")
        out.append("-" * 40)
        out.append(_preview_lang(full_data_dir))
    else:
        out.append(f"\n  Data directory not found or not configured: {full_data_dir}")

    # Baseline results
    out.append("\n" + "-" * 40)
    out.append("BASELINE RESULTS")
    out.append("-" * 40)
    out.append(_preview_results(full_recipe_dir))

    result = "\n".join(out)

    # Truncate if too long
    if len(result) > 6000:
        result = result[:6000] + "\n... (truncated)"

    return result
