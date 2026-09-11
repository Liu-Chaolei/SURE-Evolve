from pathlib import Path

from omegaconf import DictConfig, OmegaConf


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG_PATH = _PACKAGE_ROOT / "config" / "default.yaml"


def merge_with_default_survey_config(config: DictConfig) -> DictConfig:
    """Merge a survey preset config on top of the bundled default survey config."""
    default_root = OmegaConf.load(_DEFAULT_CONFIG_PATH)
    default_survey = OmegaConf.select(default_root, "survey") or default_root
    current_survey = OmegaConf.select(config, "survey")
    survey_config = current_survey if current_survey is not None else config
    return OmegaConf.merge(default_survey, survey_config)


def resolve_repo_relative_path(path_str: str) -> str:
    """Resolve a config path relative to the integrated SurveyAgent package root."""
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return str(path)
    return str((_PACKAGE_ROOT / path).resolve())
