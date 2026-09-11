"""Task-specific integrity checks extracted from the shared executor."""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any
from ..core.utils.candidate_type import ARCH, FINE_TUNE, normalize_candidate_type
from ..core.utils.candidate_changes import validate_arch_candidate_changes

ASR_TRAINING_CANDIDATE_TYPES = {ARCH, FINE_TUNE}
ASR_FATAL_TRAIN_PATTERNS = (
    "npu out of memory",
    "acl error: 207001",
    "cuda out of memory",
    "torch.cuda.outofmemoryerror",
    "cuda error: out of memory",
    "cublas_status_alloc_failed",
)
ASR_ARCH_CODE_MARKERS = (
    "--num-encoder-layers",
    "--encoder-dim",
    "--feedforward-dim",
    "--encoder-unmasked-dim",
)


class GuardContext:
    def __init__(self, exp):
        self.exp = exp

    def __getattr__(self, name):
        return getattr(self.exp, name)


class AsrGuards(GuardContext):
    def validate(self, role_paths: dict) -> list[str]:
        errors: list[str] = []
        errors.extend(self._candidate_changes_guard_errors())
        errors.extend(self._architecture_guard_errors())
        errors.extend(self._asr_duration_guard_errors())
        errors.extend(self._asr_checkpoint_guard_errors())
        errors.extend(self._asr_hyp_format_guard_errors(role_paths))

        ref_path = self._resolve_role_path("ref", role_paths)
        hyp_path = self._resolve_role_path("hyp", role_paths)
        if (
            ref_path is None
            or hyp_path is None
            or not ref_path.exists()
            or not hyp_path.exists()
        ):
            return errors

        refs = self._read_key_text_file(ref_path)
        hyps = self._read_key_text_file(hyp_path)
        if len(refs) < 100 or not hyps:
            return errors
        shared_keys = [key for key in refs if key in hyps]
        if len(shared_keys) < max(100, int(len(refs) * 0.5)):
            return errors

        exact_matches = 0
        normalized_matches = 0
        for key in shared_keys:
            if hyps[key] == refs[key]:
                exact_matches += 1
            if self._normalize_text(hyps[key]) == self._normalize_text(refs[key]):
                normalized_matches += 1

        coverage = len(shared_keys) / len(refs)
        normalized_ratio = normalized_matches / len(shared_keys)
        if coverage >= 0.95 and normalized_ratio >= 0.95:
            errors.append(
                "ASR hypothesis appears to copy the reference transcript ("
                f"{normalized_matches}/{len(shared_keys)} normalized matches, "
                f"{exact_matches}/{len(shared_keys)} exact matches, "
                f"{len(shared_keys)}/{len(refs)} keys covered). Generate hypotheses "
                "from model decoding, not from ref files or supervision manifests."
            )
            return errors

        normalized_hyp_counts: dict[str, int] = {}
        for key in shared_keys:
            normalized = self._normalize_text(hyps[key])
            normalized_hyp_counts[normalized] = (
                normalized_hyp_counts.get(normalized, 0) + 1
            )

        if len(shared_keys) >= 100 and coverage >= 0.95:
            unique_ratio = len(normalized_hyp_counts) / len(shared_keys)
            most_common = max(normalized_hyp_counts.values(), default=0)
            most_common_ratio = most_common / len(shared_keys)
            if unique_ratio < 0.01 or most_common_ratio > 0.5:
                errors.append(
                    "ASR hypothesis has implausibly low transcript diversity ("
                    f"{len(normalized_hyp_counts)} unique normalized hypotheses over "
                    f"{len(shared_keys)} matched keys; most common covers "
                    f"{most_common}/{len(shared_keys)} keys). Generate hypotheses "
                    "from real model decoding, not a constant placeholder or fallback token."
                )
        return errors

    def _asr_checkpoint_guard_errors(self) -> list[str]:
        if (
            normalize_candidate_type(self.candidate_type_hint)
            not in ASR_TRAINING_CANDIDATE_TYPES
        ):
            return []
        if self._is_official_asr_baseline_draft():
            return []

        allow_baseline = self._execution_env_truthy(
            "SURE_ASR_ALLOW_BASELINE_CHECKPOINT", False
        )
        require_local = (not allow_baseline) and self._execution_env_truthy(
            "SURE_ASR_REQUIRE_LOCAL_CHECKPOINT",
            True,
        )
        forbid_baseline = (not allow_baseline) and self._execution_env_truthy(
            "SURE_ASR_FORBID_BASELINE_CHECKPOINT_FOR_TRAINING",
            True,
        )
        fail_on_fatal = self._execution_env_truthy(
            "SURE_ASR_FAIL_ON_TRAIN_FATAL_WITHOUT_CHECKPOINT",
            True,
        )
        if not (require_local or forbid_baseline or fail_on_fatal):
            return []

        workspace = Path(self.workspace_path).resolve()
        baseline_root = self._execution_env_path("SURE_BASELINE_CHECKPOINT_DIR")
        if baseline_root and baseline_root.exists():
            baseline_root = baseline_root.resolve(strict=False)
        else:
            baseline_root = None

        model_root = workspace / "models"
        checkpoints = (
            [
                path
                for path in sorted(model_root.rglob("epoch-*.pt"))
                if path.name and not path.name.startswith("bad-model-")
            ]
            if model_root.exists()
            else []
        )
        bad_checkpoints = (
            [path for path in sorted(model_root.rglob("bad-model-*.pt"))]
            if model_root.exists()
            else []
        )

        baseline_checkpoints: list[Path] = []
        external_checkpoints: list[Path] = []
        valid_local_checkpoints: list[Path] = []
        for checkpoint in checkpoints:
            resolved = checkpoint.resolve(strict=False)
            is_baseline = bool(
                baseline_root
                and (
                    self._path_is_relative_to(resolved, baseline_root)
                    or self._checkpoint_matches_baseline_copy(checkpoint, baseline_root)
                )
            )
            if is_baseline:
                baseline_checkpoints.append(checkpoint)
                continue
            if self._path_is_relative_to(resolved, workspace):
                if (
                    self._candidate_started_at <= 0
                    or self._path_mtime(checkpoint) >= self._candidate_started_at - 1
                ):
                    valid_local_checkpoints.append(checkpoint)
                else:
                    external_checkpoints.append(checkpoint)
                continue
            external_checkpoints.append(checkpoint)

        errors: list[str] = []
        fatal_sources = self._asr_fatal_train_log_paths(workspace)
        if self._text_has_asr_train_fatal(self.terminal_output):
            fatal_sources.append(Path("terminal output"))
        if fail_on_fatal and fatal_sources:
            latest_fatal = max(
                (
                    self._path_mtime(path)
                    for path in fatal_sources
                    if path.name != "terminal output"
                ),
                default=self._candidate_started_at or 0.0,
            )
            has_later_checkpoint = any(
                self._path_mtime(path) >= latest_fatal
                for path in valid_local_checkpoints
            )
            if not has_later_checkpoint:
                source_text = ", ".join(
                    self._workspace_relative(path, workspace)
                    if path.name != "terminal output"
                    else "terminal output"
                    for path in fatal_sources[:3]
                )
                errors.append(
                    "ASR training candidate observed a fatal CUDA/CUBLAS training failure "
                    "without a later valid workspace-local checkpoint. Fatal source(s): "
                    f"{source_text}."
                )

        details = self._asr_checkpoint_guard_details(
            baseline_checkpoints=baseline_checkpoints,
            external_checkpoints=external_checkpoints,
            bad_checkpoints=bad_checkpoints,
        )
        if require_local and not valid_local_checkpoints:
            errors.append(
                "ASR training candidate did not produce a workspace-local epoch checkpoint. "
                "Metric is skipped to avoid scoring a baseline fallback or failed training artifact."
                + details
            )
        if forbid_baseline and baseline_checkpoints and not valid_local_checkpoints:
            errors.append(
                "ASR training candidate checkpoint resolves inside SURE_BASELINE_CHECKPOINT_DIR "
                "and no workspace-local candidate checkpoint was found. Decode must use a checkpoint "
                "produced by this candidate, not the official baseline." + details
            )
        return errors

    def _asr_duration_guard_errors(self) -> list[str]:
        if (
            normalize_candidate_type(self.candidate_type_hint)
            not in ASR_TRAINING_CANDIDATE_TYPES
        ):
            return []
        if self._is_official_asr_baseline_draft():
            return []
        floor = self._asr_min_train_duration()
        if floor <= 1:
            return []
        path = Path(self.workspace_path) / "artifacts" / "candidate_changes.json"
        if not path.is_file():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        declared = self._asr_candidate_declared_durations(payload)
        below = [value for value in declared if 0 < value < floor]
        if not below:
            return []
        return [
            "ASR training candidate used max-duration below the configured floor "
            f"({min(below)} < {floor}). Do not recover SURE_MAX_DURATION from failed "
            "runtime helper logs or tracebacks; only use a successful helper result "
            "or an explicitly configured fixed value."
        ]

    def _asr_min_train_duration(self) -> int:
        values: list[int] = []
        for name in ("SURE_TRAIN_DURATION_MIN", "SURE_DURATION_AUTOTUNE_MIN"):
            raw = self.execution_env.get(name, os.environ.get(name))
            value = self._parse_positive_int(raw)
            if value > 0:
                values.append(value)
        return max(values) if values else 100

    def _asr_candidate_declared_durations(self, payload: Any) -> list[int]:
        duration_keys = {
            "max_duration",
            "actual_train_max_duration",
            "resolved_max_duration",
            "selected_max_duration",
            "selected_duration",
            "selected_train_duration",
            "max_duration_bound",
            "resolved_max_duration_bound",
            "resolved_or_fixed_max_duration_bound",
        }
        values: list[int] = []

        def visit(node: Any) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    if str(key) in duration_keys:
                        parsed = self._parse_positive_int(value)
                        if parsed > 0:
                            values.append(parsed)
                    visit(value)
            elif isinstance(node, list):
                for item in node:
                    visit(item)

        visit(payload)
        return values

    @staticmethod
    def _text_has_asr_train_fatal(text: str) -> bool:
        lowered = str(text or "").lower()
        return any(marker in lowered for marker in ASR_FATAL_TRAIN_PATTERNS)

    def _asr_fatal_train_log_paths(self, workspace: Path) -> list[Path]:
        log_root = workspace / "working"
        if not log_root.exists():
            return []
        result: list[Path] = []
        for path in sorted(log_root.rglob("*.log")):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if self._text_has_asr_train_fatal(text):
                result.append(path)
        return result

    def _asr_checkpoint_guard_details(
        self,
        *,
        baseline_checkpoints: list[Path],
        external_checkpoints: list[Path],
        bad_checkpoints: list[Path],
    ) -> str:
        workspace = Path(self.workspace_path).resolve()
        parts: list[str] = []
        if baseline_checkpoints:
            parts.append(
                "baseline checkpoint(s): "
                + ", ".join(
                    self._workspace_relative(path, workspace)
                    for path in baseline_checkpoints[:3]
                )
            )
        if external_checkpoints:
            parts.append(
                "external/stale checkpoint(s): "
                + ", ".join(
                    self._workspace_relative(path, workspace)
                    for path in external_checkpoints[:3]
                )
            )
        if bad_checkpoints:
            parts.append(
                "bad checkpoint(s): "
                + ", ".join(
                    self._workspace_relative(path, workspace)
                    for path in bad_checkpoints[:3]
                )
            )
        return " " + "; ".join(parts) if parts else ""

    def _is_official_asr_baseline_draft(self) -> bool:
        if self.stage != "draft":
            return False
        path = Path(self.workspace_path) / "artifacts" / "official_baseline.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return str(payload.get("baseline_type") or "").strip().lower() == "official"

    def _asr_hyp_format_guard_errors(
        self, role_paths: dict[str, str | None]
    ) -> list[str]:
        hyp_path = self._resolve_role_path("hyp", role_paths)
        if hyp_path is None or not hyp_path.exists():
            return []
        errors: list[str] = []
        try:
            lines = hyp_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError as exc:
            return [f"ASR hypothesis file could not be read: {exc}"]
        checked = 0
        for lineno, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            checked += 1
            if "\t" not in line:
                errors.append(
                    f"ASR hypothesis line {lineno} is not key-tab-text format; "
                    "write artifacts/hyp.txt as utterance_id<TAB>hypothesis."
                )
                break
            key, text = line.split("\t", 1)
            if not key.strip():
                errors.append(
                    f"ASR hypothesis line {lineno} has an empty utterance id."
                )
                break
            # A real ASR decoder can emit a blank hypothesis for short or difficult
            # utterances. Treat that as all-token deletion during WER scoring rather
            # than as an artifact-format failure; low-diversity guards below still
            # reject constant blank/placeholder outputs.
        if lines and checked == 0:
            errors.append(
                "ASR hypothesis file contains no non-empty key-tab-text rows."
            )
        return errors

    def _architecture_guard_errors(self):
        changes = Path(self.workspace_path) / "artifacts/candidate_changes.json"
        if changes.exists() and json.loads(changes.read_text()).get("recipe_changes"):
            return []
        if normalize_candidate_type(self.candidate_type_hint) == ARCH and not any(
            marker in (self.code or "") + (self.terminal_output or "")
            for marker in ASR_ARCH_CODE_MARKERS
        ):
            return [
                "ASR architecture candidate did not change a Zipformer structure parameter"
            ]
        return []


class TtsGuards(GuardContext):
    def validate(self, roles):
        errors = self._tts_artifact_guard_errors(roles)
        errors.extend(self._candidate_changes_guard_errors())
        if normalize_candidate_type(self.candidate_type_hint) == ARCH:
            errors.extend(validate_arch_candidate_changes(self.workspace_path))
        return errors

    def _tts_artifact_guard_errors(
        self, role_paths: dict[str, str | None]
    ) -> list[str]:
        if "samples_jsonl" not in self.task_card.required_roles:
            return []
        samples_path = self._resolve_role_path("samples_jsonl", role_paths)
        if samples_path is None or not samples_path.exists():
            return []

        rows: list[dict[str, Any]] = []
        errors: list[str] = []
        try:
            with samples_path.open("r", encoding="utf-8") as f:
                for lineno, line in enumerate(f, start=1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        errors.append(
                            f"TTS samples_jsonl is invalid JSON on line {lineno}: {exc.msg}."
                        )
                        continue
                    if not isinstance(row, dict):
                        errors.append(
                            f"TTS samples_jsonl line {lineno} is not a JSON object."
                        )
                        continue
                    row["_line_no"] = lineno
                    rows.append(row)
        except OSError as exc:
            return [f"TTS samples_jsonl could not be read: {exc}"]

        if not rows:
            errors.append("TTS samples_jsonl contains no rows.")
            return errors

        prediction_paths: list[Path] = []
        for row in rows:
            lineno = row.get("_line_no", "?")
            prediction_value = str(row.get("prediction_audio") or "")
            prediction_path = self._resolve_samples_jsonl_path(
                samples_path, prediction_value
            )
            prediction_paths.append(prediction_path)
            try:
                if (
                    not prediction_path.is_file()
                    or prediction_path.stat().st_size < 1024
                ):
                    errors.append(
                        f"TTS prediction_audio on line {lineno} is empty or implausibly small: {prediction_path}"
                    )
            except OSError:
                errors.append(
                    f"TTS prediction_audio on line {lineno} is empty or implausibly small: {prediction_path}"
                )
            reference_value = str(row.get("reference_audio") or "")
            if reference_value:
                reference_path = self._resolve_samples_jsonl_path(
                    samples_path, reference_value
                )
                try:
                    if prediction_path.samefile(reference_path):
                        errors.append(
                            f"TTS prediction_audio on line {lineno} points to the reference_audio path: {prediction_path}"
                        )
                        continue
                except OSError:
                    pass
                if self._same_file_content(prediction_path, reference_path):
                    errors.append(
                        f"TTS prediction_audio on line {lineno} is byte-identical to reference_audio: {reference_path}"
                    )

        resolved_prediction_paths = {
            str(path.resolve(strict=False)) for path in prediction_paths
        }
        if len(rows) > 1 and len(resolved_prediction_paths) == 1:
            errors.append(
                "TTS samples_jsonl reuses the same prediction_audio for every sample; "
                "generate a distinct audio file for each target text."
            )
        return errors
