"""Task semantics behind the ordinary controller. Importable without Torch."""

from __future__ import annotations

import gzip
import json
import os
import sys
from pathlib import Path
from typing import Any, Protocol

from ..core.artifacts import publish_bundle
from ..core.datasets import DatasetSplitSpec, split_specs, validate_split_groups

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TaskAdapter(Protocol):
    name: str
    task: str

    def environment(self, sure: dict) -> dict[str, str]: ...
    def preflight(self, sure: dict) -> dict: ...
    def collect_model(self, workspace: str, env: dict[str, str]) -> dict: ...
    def validate_outputs(self, exp: Any, roles: dict) -> list[str]: ...
    def phase_environment(self, split: DatasetSplitSpec) -> dict[str, str]: ...
    def context(self) -> dict: ...
    def baseline_candidate_type(self, sure: dict, remote: bool = False) -> str: ...
    def frozen_code(self, artifact: str) -> str: ...
    def scoring_roles(self, workspace: str, env: dict, roles: dict) -> dict: ...
    def final_candidates(
        self, controller: Any, baseline: dict, candidates: list[dict]
    ) -> tuple[dict, list[dict]]: ...
    def execute_candidate(
        self,
        action: str,
        parameters: dict,
        settings: dict,
        manifest: dict,
        resources: dict,
        parent: str,
        frozen: str,
    ) -> None: ...


class BaseAdapter:
    name = ""
    task = ""
    modules: tuple[str, ...] = ()
    required_resources: tuple[str, ...] = ()

    def baseline_candidate_type(self, sure: dict, remote: bool = False) -> str:
        return "inference"

    def final_candidates(
        self, controller: Any, baseline: dict, candidates: list[dict]
    ) -> tuple[dict, list[dict]]:
        if (controller.sure_config.get("full_training") or {}).get("enabled"):
            raise ValueError(
                f"Optional full_training is not implemented for {self.name}"
            )
        return baseline, candidates

    def execute_candidate(
        self, action, parameters, settings, manifest, resources, parent, frozen
    ):
        raise NotImplementedError(f"No worker implementation for {self.name}")

    def environment(self, sure: dict) -> dict[str, str]:
        runtime = sure.get("runtime") or {}
        settings = dict(sure.get("task") or {})
        settings.setdefault("training", {})
        settings.setdefault("inference", {})
        settings.setdefault("resources", {})
        legacy_env = sure.get("execution_env") or {}
        legacy_key = {"asr": "SURE_ICEFALL_PYTHON", "tts": "SURE_TTS_PYTHON"}.get(
            self.task, "SURE_WORKER_PYTHON"
        )
        worker = str(
            runtime.get("python")
            or legacy_env.get("SURE_WORKER_PYTHON")
            or legacy_env.get(legacy_key)
            or sys.executable
        )
        env = {
            "SURE_TASK_ADAPTER": self.name,
            "SURE_WORKER_PYTHON": worker,
            "SURE_TASK_SETTINGS": json.dumps(settings),
            "SURE_TASK_WRAPPER": str(
                PROJECT_ROOT / "playground/sure_master/tools/run_task_candidate.py"
            ),
        }
        specs = split_specs(sure)
        if "search" in specs:
            env.update(self.phase_environment(specs["search"]))
        for split in ("train", "train_validation"):
            if split in specs and specs[split].manifest:
                env[f"SURE_{split.upper()}_MANIFEST"] = specs[split].manifest
        return env

    def phase_environment(self, split: DatasetSplitSpec) -> dict[str, str]:
        env = {str(k): str(v) for k, v in split.execution_env.items()}
        if split.manifest:
            env["SURE_EVAL_MANIFEST"] = split.manifest
        return env

    def preflight(self, sure: dict) -> dict:
        profiles = sure.get("base_models") or {}
        profile = profiles.get(sure.get("task_id")) or profiles.get(self.task) or {}
        for role, value in (profile.get("source_paths") or {}).items():
            if not Path(value).exists():
                raise FileNotFoundError(f"Missing model source {role}: {value}")
        resources = sure.get("task", {}).get("resources") or {}
        for role in self.required_resources:
            if not resources.get(role):
                raise ValueError(f"Missing {self.name} resource configuration: {role}")
        for role, value in resources.items():
            if not value or not Path(value).exists():
                raise FileNotFoundError(f"Missing model resource {role}: {value}")
        specs = split_specs(sure)
        reports = [spec.validate() for spec in specs.values()]
        validate_split_groups(specs)
        for split, spec in specs.items():
            if spec.manifest and Path(spec.manifest).suffix == ".jsonl":
                for line in Path(spec.manifest).read_text().splitlines():
                    row = json.loads(line)
                    for key in ("audio", "reference_audio"):
                        if row.get(key) and not Path(row[key]).is_file():
                            raise FileNotFoundError(
                                f"Missing {split} audio: {row[key]}"
                            )
        return {"adapter": self.name, "splits": reports, "modules": list(self.modules)}

    def collect_model(self, workspace: str, env: dict[str, str]) -> dict:
        root = Path(workspace)
        resource_file = root / "artifacts/model_resources.json"
        if not resource_file.is_file():
            if env.get("SURE_REQUIRE_MODEL_ARTIFACT") == "1":
                raise FileNotFoundError(f"Missing replay resources: {resource_file}")
            return {}
        payload = json.loads(resource_file.read_text())
        return publish_bundle(
            root,
            self.name,
            payload["resources"],
            model_config=payload.get("model_config"),
            inference_config=payload.get("inference_config"),
            provenance=payload.get("provenance"),
        )

    def scoring_roles(self, workspace: str, env: dict, roles: dict) -> dict:
        return roles

    def frozen_code(self, artifact: str) -> str:
        return (
            "import os, subprocess\nfrom pathlib import Path\n"
            "def main():\n"
            "    subprocess.run([os.environ['SURE_WORKER_PYTHON'], "
            "os.environ['SURE_TASK_WRAPPER'], '--action', 'infer', '--model-artifact', "
            + repr(artifact)
            + "], check=True)\n"
            "if __name__ == '__main__':\n    main()\n"
        )

    def context(self) -> dict:
        return {
            "adapter": self.name,
            "candidate_types": ["inference", "fine_tune", "arch"],
            "candidate_entrypoint": "SURE_TASK_WRAPPER",
            "candidate_arguments": "--action infer|fine_tune|arch --parameters-json '{...}'",
            "initialization": "inference: round best; training: fixed baseline, fresh optimizer",
            "frozen_evaluation": "inference only; never train or change saved parameters",
        }


class AsrAdapter(BaseAdapter):
    name, task = "asr.zipformer", "asr"
    modules = ("torch", "k2", "lhotse", "sentencepiece")

    def baseline_candidate_type(self, sure: dict, remote: bool = False) -> str:
        env = sure.get("execution_env") or {}
        trains = (
            sure.get("execution_mode") == "slurm"
            or env.get("SURE_BASELINE_USE_PRETRAINED") == "0"
        )
        return "fine_tune" if trains or remote else "inference"

    def final_candidates(self, controller, baseline, candidates):
        from ..core.full_training import retrain_selected

        return retrain_selected(controller, baseline, candidates)

    def execute_candidate(
        self, action, parameters, settings, manifest, resources, parent, frozen
    ):
        from .zipformer import execute

        execute(action, parameters, parent, frozen)

    def context(self) -> dict:
        return {
            **super().context(),
            "initialization": "training: from scratch, fixed seed/data/epoch budget; inference: frozen round parent",
            "candidate_parameters": {
                "train_args_json": "list of Zipformer train.py flags; budget, output paths and world size are fixed",
                "decode_args_json": "list of Zipformer decode.py flags",
                "decode_method": "recipe decoding method",
            },
        }

    def environment(self, sure: dict) -> dict[str, str]:
        env = super().environment(sure)
        env["SURE_ICEFALL_PYTHON"] = (
            os.environ.get("SURE_REMOTE_ICEFALL_PYTHON") or env["SURE_WORKER_PYTHON"]
        )
        if os.environ.get("SURE_REMOTE_ICEFALL_PYTHON"):
            env["SURE_REMOTE_ICEFALL_PYTHON"] = os.environ["SURE_REMOTE_ICEFALL_PYTHON"]
        if os.environ.get("SURE_LOCAL_ICEFALL_PYTHON"):
            env["SURE_LOCAL_ICEFALL_PYTHON"] = os.environ["SURE_LOCAL_ICEFALL_PYTHON"]
        backend = (sure.get("runtime") or {}).get("accelerator")
        if backend in {"cpu", "npu"}:
            size = (
                1
                if backend == "cpu"
                else int((sure.get("runtime") or {}).get("world_size", 1))
            )
            env.update(
                SURE_USE_FP16="0",
                ASR_WORLD_SIZE=str(size),
                SURE_BASELINE_WORLD_SIZE=str(size),
            )
        if backend == "npu":
            env.update(SURE_DECODE_METHOD="greedy_search", SURE_DURATION_AUTOTUNE="0")
        return env

    def phase_environment(self, split: DatasetSplitSpec) -> dict[str, str]:
        return {
            **super().phase_environment(split),
            "SURE_ASR_EVAL_SPLITS": "test" if split.name == "holdout" else "dev",
        }

    def preflight(self, sure: dict) -> dict:
        report = super().preflight(sure)
        profile = sure["base_models"].get(sure["task_id"], {})
        sources = profile.get("source_paths") or {}
        data = Path(sources["data"])
        if not (data / "lang_bpe_500/bpe.model").is_file():
            raise FileNotFoundError("ASR BPE model missing")
        if (sure.get("execution_env") or {}).get("SURE_ASR_DATASET") == "tedlium3":
            for split in ("train", "dev", "test"):
                with gzip.open(
                    data / "fbank" / f"tedlium_cuts_{split}.jsonl.gz", "rt"
                ) as stream:
                    if not stream.readline().strip():
                        raise ValueError(f"Empty {split} feature manifest")
        if sure.get("execution_mode") == "slurm":
            preparation = json.loads((data / "preparation.json").read_text())
            expected_hours = float(sure.get("execution_contract", {}).get("training_hours", 100))
            if (not preparation.get("features_ready") or preparation.get("seed") != 42
                    or abs(float(preparation.get("hours", 0)) - expected_hours) > 0.02):
                raise ValueError("Search subset must match the fixed duration and seed=42")
            import tempfile
            from ..core.utils.metric import SureMetricRunner
            from ..core.utils.task_cards import resolve_task_card

            card = resolve_task_card(sure["task_cards_path"], sure["task_id"])
            with tempfile.TemporaryDirectory(prefix="sure-asr-wer-") as directory:
                fixture = Path(directory)
                (fixture / "ref.txt").write_text("check\tTHE CAT SAT\n")
                (fixture / "hyp.txt").write_text("check\tTHE DOG SAT\n")
                result = SureMetricRunner(
                    sure["root"], sure.get("pythonpath"), device="cpu"
                ).run(
                    card,
                    fixture,
                    fixture / "metric",
                    {"ref": str(fixture / "ref.txt"), "hyp": str(fixture / "hyp.txt")},
                )
                if not result.success or abs(result.score - 1 / 3) > 1e-8:
                    raise RuntimeError("SURE WER preflight failed")
            report["metric_fixture"] = "passed"
            full = sure.get("full_training", {}).get("data")
            if full:
                from ..core.artifacts import file_digest

                if file_digest(data / "lang_bpe_500/bpe.model") != file_digest(
                    Path(full) / "lang_bpe_500/bpe.model"
                ):
                    raise ValueError("Search and full training must share the same BPE")
            refs = [
                sure["inputs"]["ref"],
                sure["final_evaluation"]["selection_ref"],
                sure["final_evaluation"]["test_ref"],
            ]
            groups = []
            for ref in refs:
                lines = Path(ref).read_text().splitlines()
                keys = [line.split("\t", 1)[0] for line in lines if "\t" in line]
                if not keys or len(set(keys)) != len(lines):
                    raise ValueError(
                        "Malformed, duplicate or empty evaluation references"
                    )
                groups.append({key.rsplit("-", 1)[0] for key in keys})
            if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
                raise ValueError("Search, selection and test contain overlapping talks")
        return report

    def collect_model(self, workspace: str, env: dict[str, str]) -> dict:
        from ..core.utils.model_artifact import retain_model_artifact

        retained = retain_model_artifact(workspace)
        if not retained:
            if env.get("SURE_REQUIRE_MODEL_ARTIFACT") == "1":
                raise RuntimeError("ASR did not produce a replayable model")
            return {}
        legacy = Path(retained["model_artifact"]).parent
        manifest = json.loads((legacy / "manifest.json").read_text())
        changes = legacy / "artifacts/candidate_changes.json"
        payload = json.loads(changes.read_text()) if changes.is_file() else {}
        if not payload:
            baseline = legacy / "artifacts/official_baseline.json"
            if baseline.is_file():
                raw = json.loads(baseline.read_text())
                payload = {
                    "inference_config": {
                        "decoding_method": raw.get("decoding_method"),
                        "decode_avg": raw.get("avg", 1),
                        "use_averaged_model": raw.get("use_averaged_model", False),
                    }
                }
        return publish_bundle(
            Path(workspace),
            self.name,
            {
                "checkpoint": legacy / manifest["checkpoint"],
                "tokenizer": legacy / manifest["bpe_model"],
            },
            model_config=payload.get("arch_config"),
            inference_config=payload.get("inference_config"),
            provenance=payload.get("training_config"),
            legacy_root=legacy,
        )

    def validate_outputs(self, exp: Any, roles: dict) -> list[str]:
        from .guards import AsrGuards

        return AsrGuards(exp).validate(roles)


class TtsAdapter(BaseAdapter):
    name, task = "tts.f5tts", "tts"
    modules = ("torch", "torchaudio", "f5_tts", "vocos", "accelerate")
    required_resources = ("source", "checkpoint", "vocab", "vocoder")

    def execute_candidate(
        self, action, parameters, settings, manifest, resources, parent, frozen
    ):
        from ..runtime.accelerator import RuntimeBackend
        from .f5tts import execute

        backend = RuntimeBackend()
        backend.seed(int(settings.get("training", {}).get("seed", 42)))
        execute(action, parameters, settings, manifest, resources, backend, parent)

    def context(self) -> dict:
        from .f5tts import INFERENCE_KEYS, TRAIN_KEYS

        return {
            **super().context(),
            "candidate_parameters": {
                "inference": sorted(INFERENCE_KEYS),
                "training": sorted(TRAIN_KEYS),
                "architecture": {
                    "depth": [18, 20, 22, 24],
                    "ff_mult": [2, 3, 4],
                    "conv_layers": [2, 4, 6],
                    "qk_norm": ["none", "rms_norm"],
                    "attn_mask_enabled": [False, True],
                    "checkpoint_activations": [False, True],
                },
            },
        }

    def environment(self, sure: dict) -> dict[str, str]:
        env = super().environment(sure)
        env["SURE_TTS_PYTHON"] = env["SURE_WORKER_PYTHON"]
        return env

    def validate_outputs(self, exp: Any, roles: dict) -> list[str]:
        from .guards import TtsGuards

        return TtsGuards(exp).validate(roles)


class SdAdapter(BaseAdapter):
    name, task = "sd.diarizen", "sd"
    modules = ("torch", "torchaudio", "diarizen", "pyannote.audio", "toml")
    required_resources = ("source", "model", "embedding")

    def execute_candidate(
        self, action, parameters, settings, manifest, resources, parent, frozen
    ):
        from ..runtime.accelerator import RuntimeBackend
        from .diarization import execute

        backend = RuntimeBackend()
        backend.seed(int(settings.get("training", {}).get("seed", 42)))
        execute(action, parameters, settings, manifest, resources, backend, parent)

    def context(self) -> dict:
        from .diarization import INFER_KEYS, TRAIN_KEYS, ARCH_VALUES

        return {
            **super().context(),
            "candidate_parameters": {
                "inference": sorted(INFER_KEYS),
                "training": sorted(TRAIN_KEYS),
                "architecture": {k: sorted(v) for k, v in ARCH_VALUES.items()},
            },
            "metric_protocol": {
                "collar": 0.25,
                "aggregation": "session_mean_error_rate",
                "uem": "trusted clipping",
            },
        }

    def scoring_roles(self, workspace: str, env: dict, roles: dict) -> dict:
        from .diarization import clip_rttm

        root = Path(workspace)
        manifest = Path(env["SURE_EVAL_MANIFEST"])
        result = dict(roles)
        for role in ("ref", "hyp"):
            raw = Path(roles[role])
            source = raw if raw.is_absolute() else root / raw
            target = root / "metric" / f"scoring_{role}.rttm"
            clip_rttm(source, manifest, target)
            result[role] = str(target)
        return result

    def validate_outputs(self, exp: Any, roles: dict) -> list[str]:
        from .diarization import validate_rttm_outputs

        errors = exp._candidate_changes_guard_errors()
        manifest = exp.execution_env.get("SURE_EVAL_MANIFEST", "")
        try:
            validate_rttm_outputs(
                Path(exp.workspace_path) / "artifacts/hyp.rttm", Path(manifest)
            )
        except (ValueError, OSError) as exc:
            errors.append(str(exc))
        return errors


_ADAPTERS = {item.name: item() for item in (AsrAdapter, TtsAdapter, SdAdapter)}
_DEFAULTS = {"asr": "asr.zipformer", "tts": "tts.f5tts", "sd": "sd.diarizen"}


def get_adapter(task: str, name: str | None = None) -> BaseAdapter:
    selected = name or _DEFAULTS.get(task, task)
    if selected not in _ADAPTERS:
        raise ValueError(f"No task adapter registered for {selected!r}")
    adapter = _ADAPTERS[selected]
    if task in _DEFAULTS and adapter.task != task:
        raise ValueError(f"Adapter {selected} cannot execute {task}")
    return adapter


def register_adapter(adapter: TaskAdapter) -> None:
    if not adapter.name or adapter.name in _ADAPTERS:
        raise ValueError(f"Empty or duplicate task adapter: {adapter.name}")
    _ADAPTERS[adapter.name] = adapter
    _DEFAULTS.setdefault(adapter.task, adapter.name)
