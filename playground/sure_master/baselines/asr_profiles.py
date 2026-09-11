from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class AsrDatasetProfile:
    name: str
    display_name: str
    default_eval_splits: tuple[str, ...]
    allowed_eval_splits: tuple[str, ...]
    search_splits: tuple[str, ...]
    final_splits: tuple[str, ...]
    manifest_patterns: dict[str, str]
    ref_file_prefix: str
    key_normalization: str
    grouping_key: str


@dataclass(frozen=True)
class AsrRecipeProfile:
    name: str
    model_family: str
    dataset: str
    recipe_family: str
    arch_args: tuple[str, ...]
    train_only_args: tuple[str, ...]
    train_data_args: tuple[str, ...]
    decode_data_args: tuple[str, ...]
    default_bpe_model: Path
    official_bpe_model: Path | None
    official_model_id: str | None
    official_model_url: str | None
    default_checkpoint_dir_env: str | None
    checkpoint_compatibility: str
    decode_patch_strategy: str


LIBRISPEECH_OFFICIAL_MODEL_ID = (
    "Zengwei/icefall-asr-librispeech-zipformer-large-transducer-with-CR-CTC-20241019"
)

DATASET_PROFILES: dict[str, AsrDatasetProfile] = {
    "librispeech": AsrDatasetProfile(
        name="librispeech",
        display_name="LibriSpeech",
        default_eval_splits=("test-clean", "test-other"),
        allowed_eval_splits=("dev-clean", "dev-other", "test-clean", "test-other"),
        search_splits=("dev-clean", "dev-other"),
        final_splits=("test-clean", "test-other"),
        manifest_patterns={
            "dev-clean": "librispeech_cuts_dev-clean.jsonl.gz",
            "dev-other": "librispeech_cuts_dev-other.jsonl.gz",
            "test-clean": "librispeech_cuts_test-clean.jsonl.gz",
            "test-other": "librispeech_cuts_test-other.jsonl.gz",
        },
        ref_file_prefix="asr_librispeech",
        key_normalization="librispeech_strip_trailing_numeric_suffix",
        grouping_key="speaker",
    ),
    "tedlium3": AsrDatasetProfile(
        name="tedlium3",
        display_name="TEDLIUM release 3",
        default_eval_splits=("dev",),
        allowed_eval_splits=("dev", "test"),
        search_splits=("dev",),
        final_splits=("test",),
        manifest_patterns={
            "train": "tedlium_cuts_train.jsonl.gz",
            "dev": "tedlium_cuts_dev.jsonl.gz",
            "test": "tedlium_cuts_test.jsonl.gz",
        },
        ref_file_prefix="asr_tedlium3",
        key_normalization="identity",
        grouping_key="recording",
    ),
}

RECIPE_PROFILES: dict[str, AsrRecipeProfile] = {
    "librispeech_zipformer_large_cr_ctc_rnnt": AsrRecipeProfile(
        name="librispeech_zipformer_large_cr_ctc_rnnt",
        model_family="zipformer",
        dataset="librispeech",
        recipe_family="zipformer_large_cr_ctc_rnnt",
        arch_args=(
            "--use-cr-ctc",
            "1",
            "--use-ctc",
            "1",
            "--use-transducer",
            "1",
            "--use-attention-decoder",
            "0",
            "--num-encoder-layers",
            "2,2,4,5,4,2",
            "--feedforward-dim",
            "512,768,1536,2048,1536,768",
            "--encoder-dim",
            "192,256,512,768,512,256",
            "--encoder-unmasked-dim",
            "192,192,256,320,256,192",
        ),
        train_only_args=(
            "--ctc-loss-scale",
            "0.1",
            "--enable-spec-aug",
            "0",
            "--cr-loss-scale",
            "0.02",
            "--time-mask-ratio",
            "2.5",
        ),
        train_data_args=("--full-libri", "1"),
        decode_data_args=("--full-libri", "1"),
        default_bpe_model=Path("data/lang_bpe_500/bpe.model"),
        official_bpe_model=Path("data/lang_bpe_500/bpe.model"),
        official_model_id=LIBRISPEECH_OFFICIAL_MODEL_ID,
        official_model_url=f"https://huggingface.co/{LIBRISPEECH_OFFICIAL_MODEL_ID}",
        default_checkpoint_dir_env="base_model/checkpoints/zipformer_large_cr_ctc_rnnt",
        checkpoint_compatibility="official_librispeech_large_cr_ctc_rnnt",
        decode_patch_strategy="librispeech_legacy_eval_splits",
    ),
    "tedlium3_zipformer": AsrRecipeProfile(
        name="tedlium3_zipformer",
        model_family="zipformer",
        dataset="tedlium3",
        recipe_family="tedlium3_zipformer",
        arch_args=(
            "--num-encoder-layers",
            "2,2,4,5,4,2",
            "--feedforward-dim",
            "512,768,1536,2048,1536,768",
            "--encoder-dim",
            "192,256,512,768,512,256",
            "--encoder-unmasked-dim",
            "192,192,256,320,256,192",
        ),
        train_only_args=("--enable-spec-aug", "0"),
        train_data_args=(),
        decode_data_args=(),
        default_bpe_model=Path("data/lang_bpe_500/bpe.model"),
        official_bpe_model=None,
        official_model_id=None,
        official_model_url=None,
        default_checkpoint_dir_env="base_model/checkpoints/tedlium3_zipformer",
        checkpoint_compatibility="tedlium3_recipe_checkpoint_or_train_from_scratch",
        decode_patch_strategy="tedlium3_eval_splits",
    ),
}

# Keep the earlier Large profile stable for existing experiments.
RECIPE_PROFILES["tedlium3_zipformer_native"] = replace(
    RECIPE_PROFILES["tedlium3_zipformer"], name="tedlium3_zipformer_native",
    recipe_family="tedlium3_zipformer_native",
    arch_args=("--num-encoder-layers", "2,2,3,4,3,2",
               "--feedforward-dim", "512,768,1024,1536,1024,768",
               "--encoder-dim", "192,256,384,512,384,256",
               "--encoder-unmasked-dim", "192,192,256,256,256,192"),
    train_only_args=("--enable-spec-aug", "1"),
)

DEFAULT_RECIPE_BY_DATASET = {
    "librispeech": "librispeech_zipformer_large_cr_ctc_rnnt",
    "tedlium3": "tedlium3_zipformer",
}


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def get_asr_dataset_profile(name: str) -> AsrDatasetProfile:
    key = _normalize_name(name)
    try:
        return DATASET_PROFILES[key]
    except KeyError as exc:
        allowed = ", ".join(sorted(DATASET_PROFILES))
        raise ValueError(f"Unknown SURE_ASR_DATASET={name!r}; allowed: {allowed}") from exc


def get_asr_recipe_profile(name: str) -> AsrRecipeProfile:
    key = _normalize_name(name)
    try:
        return RECIPE_PROFILES[key]
    except KeyError as exc:
        allowed = ", ".join(sorted(RECIPE_PROFILES))
        raise ValueError(f"Unknown SURE_ASR_RECIPE_PROFILE={name!r}; allowed: {allowed}") from exc


def current_asr_dataset_profile() -> AsrDatasetProfile:
    return get_asr_dataset_profile(os.environ.get("SURE_ASR_DATASET", "librispeech"))


def current_asr_recipe_profile(
    dataset_profile: AsrDatasetProfile | None = None,
) -> AsrRecipeProfile:
    dataset = dataset_profile or current_asr_dataset_profile()
    default_recipe = DEFAULT_RECIPE_BY_DATASET[dataset.name]
    recipe = get_asr_recipe_profile(os.environ.get("SURE_ASR_RECIPE_PROFILE", default_recipe))
    if recipe.dataset != dataset.name:
        raise ValueError(
            f"SURE_ASR_RECIPE_PROFILE={recipe.name!r} is for dataset {recipe.dataset!r}, "
            f"but SURE_ASR_DATASET={dataset.name!r}."
        )
    return recipe


def parse_eval_splits_for_profile(
    raw: str | None,
    dataset_profile: AsrDatasetProfile,
) -> list[str]:
    if raw is None or not str(raw).strip():
        return list(dataset_profile.default_eval_splits)
    splits = [
        part.strip().replace("_", "-")
        for part in str(raw).replace(",", " ").split()
        if part.strip()
    ]
    if not splits:
        return list(dataset_profile.default_eval_splits)
    allowed = set(dataset_profile.allowed_eval_splits)
    invalid = [split for split in splits if split not in allowed]
    if invalid:
        raise ValueError(
            f"Unsupported SURE_ASR_EVAL_SPLITS value(s) for {dataset_profile.display_name}: "
            f"{', '.join(invalid)}; allowed: {', '.join(dataset_profile.allowed_eval_splits)}"
        )
    return splits


def normalize_asr_cut_id(key: str, dataset_profile: AsrDatasetProfile | None = None) -> str:
    profile = dataset_profile or current_asr_dataset_profile()
    if profile.key_normalization == "librispeech_strip_trailing_numeric_suffix":
        parts = key.rsplit("-", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0]
    return key


def recipe_arch_args(recipe_profile: AsrRecipeProfile | None = None) -> list[str]:
    profile = recipe_profile or current_asr_recipe_profile()
    return list(profile.arch_args)


def recipe_train_only_args(recipe_profile: AsrRecipeProfile | None = None) -> list[str]:
    profile = recipe_profile or current_asr_recipe_profile()
    return list(profile.train_only_args)


def recipe_train_data_args(recipe_profile: AsrRecipeProfile | None = None) -> list[str]:
    profile = recipe_profile or current_asr_recipe_profile()
    return list(profile.train_data_args)


def recipe_decode_data_args(recipe_profile: AsrRecipeProfile | None = None) -> list[str]:
    profile = recipe_profile or current_asr_recipe_profile()
    return list(profile.decode_data_args)
