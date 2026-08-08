from __future__ import annotations

import ast
import re
from pathlib import Path

from .task_cards import BaseModelProfile


ASR_ZIPFORMER_DECODE_TRAIN_ONLY_ARGS = (
    "--ctc-loss-scale",
    "--cr-loss-scale",
    "--enable-spec-aug",
    "--time-mask-ratio",
)


def save_code_to_file(directory: str | Path, filename: str, code_content: str) -> Path:
    """Save generated code under an experiment workspace."""
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    file_path = target_dir / filename
    file_path.write_text(code_content, encoding="utf-8")
    return file_path


def read_code(value: str) -> tuple[str, str]:
    """Extract Python code from a model response.

    Returns:
        A tuple of (code_to_execute, raw_code). Empty code means the response did
        not contain usable Python and should be sent to the debug loop.
    """
    match = re.search(r"```(?:python)?\s*(.*?)\s*```", value, re.DOTALL)
    if match:
        value = match.group(1).strip()
    elif value.strip().startswith("<") or "<function>" in value or "<tool_call>" in value:
        return "", value

    value = re.sub(r"<function>.*?</function>", "", value, flags=re.DOTALL)
    value = re.sub(r"</?function>", "", value).strip()
    return value, value


def validate_run_sure_script(
    code_content: str,
    role_paths: dict[str, str | None] | None = None,
) -> list[str]:
    """Return errors for candidate scripts that are syntactically valid but inert.

    A comment-only Python file passes ``py_compile`` and exits with status 0, but
    it cannot satisfy the SURE artifact contract.  Keep this validation focused
    on the execution/artifact surface so simple top-level scripts remain valid.
    """
    errors: list[str] = []
    try:
        tree = ast.parse(code_content)
    except SyntaxError as exc:
        return [f"run_sure.py is not valid Python: {exc}"]

    executable_nodes = [
        node for node in tree.body
        if not (
            isinstance(node, ast.Expr)
            and isinstance(getattr(node, "value", None), ast.Constant)
            and isinstance(node.value.value, str)
        )
    ]
    if not executable_nodes:
        errors.append("run_sure.py has no executable Python statements")

    markers = _artifact_markers_from_role_paths(role_paths)
    if (
        markers
        and not any(marker in code_content for marker in markers)
        and not _references_known_artifact_producer(code_content)
    ):
        errors.append(
            "run_sure.py does not appear to reference the required output artifact(s): "
            + ", ".join(sorted(markers))
        )

    return errors


def _artifact_markers_from_role_paths(role_paths: dict[str, str | None] | None) -> set[str]:
    default_markers = {
        "artifacts/",
        "ARTIFACTS_DIR",
        "hyp.txt",
        "samples.jsonl",
        "candidate_changes.json",
    }
    if not role_paths:
        return default_markers

    markers: set[str] = set()
    input_roles = {"ref", "source", "prompt", "prompts", "input", "text", "audio"}
    for role, value in role_paths.items():
        if value is None:
            continue
        path_text = str(value).replace("\\", "/")
        if not path_text:
            continue
        is_relative = not Path(path_text).is_absolute()
        is_output_role = str(role).lower() not in input_roles
        if is_relative and (is_output_role or path_text.startswith("artifacts/")):
            markers.add(path_text)
            markers.add(Path(path_text).name)

    return markers or default_markers


def _references_known_artifact_producer(code_content: str) -> bool:
    return any(
        marker in code_content
        for marker in (
            "SURE_ASR_ZIPFORMER_WRAPPER",
            "run_icefall_zipformer_candidate.py",
            "SURE_TTS_BATCH_INFER_WRAPPER",
            "SURE_TTS_FINETUNE_WRAPPER",
            "SURE_TTS_ARCH_WRAPPER",
            "run_f5tts_batch_infer.py",
            "run_f5tts_finetune.py",
            "run_f5tts_arch_finetune.py",
        )
    )


def validate_sure_candidate_boundary(
    code_content: str,
    sure_root: str | Path,
    base_model_profile: BaseModelProfile | None = None,
    canonical_task: str | None = None,
    require_asr_wrapper: bool = False,
) -> list[str]:
    """Return boundary violations for generated candidate code.

    Candidate scripts must generate artifacts only. The framework owns SURE
    scoring, so generated code may not import SURE or reference its source tree.
    """
    errors: list[str] = []
    root_text = str(Path(sure_root))
    root_resolved = str(Path(sure_root).resolve())
    for root in {root_text, root_resolved}:
        if root and root in code_content:
            errors.append(f"references read-only SURE root: {root}")

    try:
        tree = ast.parse(code_content)
    except SyntaxError as e:
        return [f"candidate code is not valid Python: {e}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "sure_eval" or alias.name.startswith("sure_eval."):
                    errors.append(f"imports SURE module: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "sure_eval" or module.startswith("sure_eval."):
                errors.append(f"imports SURE module: {module}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
            if value == "sure_eval" or value.startswith("sure_eval."):
                errors.append(f"references SURE module string: {value}")

    if base_model_profile is not None:
        errors.extend(_validate_base_model_boundary(code_content, tree, base_model_profile))

    if _is_asr_candidate(canonical_task, base_model_profile, code_content):
        errors.extend(_validate_asr_subprocess_safety(tree))
        errors.extend(_validate_asr_cli_compatibility(tree))
        errors.extend(_validate_asr_duration_resolution_safety(code_content))
        errors.extend(_validate_asr_lhotse_key_normalization(code_content))
        errors.extend(
            _validate_asr_zipformer_wrapper_boundary(
                tree,
                code_content,
                require_asr_wrapper=require_asr_wrapper,
            )
        )
    if _is_tts_candidate(canonical_task, base_model_profile):
        errors.extend(_validate_tts_training_boundary(code_content))
        errors.extend(_validate_tts_inference_boundary(tree))

    return sorted(set(errors))


def _is_asr_candidate(
    canonical_task: str | None,
    profile: BaseModelProfile | None,
    code_content: str,
) -> bool:
    if (canonical_task or "").strip().lower() == "asr":
        return True
    if profile is not None and profile.framework.strip().lower() == "icefall":
        return True
    return "train.py" in code_content or "decode.py" in code_content


def _is_tts_candidate(
    canonical_task: str | None,
    profile: BaseModelProfile | None,
) -> bool:
    if (canonical_task or "").strip().lower() == "tts":
        return True
    if profile is not None and profile.framework.strip().lower() in {"f5-tts", "f5_tts"}:
        return True
    return False


def _validate_tts_training_boundary(code_content: str) -> list[str]:
    errors: list[str] = []
    forbidden_markers = (
        "src/f5_tts/train/finetune_cli.py",
        "src/f5_tts/train/train.py",
        "f5_tts/train/finetune_cli.py",
        "f5_tts/train/train.py",
    )
    for marker in forbidden_markers:
        if marker in code_content:
            errors.append(
                "TTS candidate calls a raw F5-TTS training entrypoint; "
                "use SURE_TTS_FINETUNE_WRAPPER or SURE_TTS_ARCH_WRAPPER instead"
            )
            break
    return errors


def _validate_tts_inference_boundary(tree: ast.AST) -> list[str]:
    errors: list[str] = []
    string_literals = _collect_string_literals(tree)
    forbidden_path_markers = (
        "src/f5_tts/infer/infer_cli.py",
        "f5_tts/infer/infer_cli.py",
    )
    direct_path = any(
        marker in value
        for value in string_literals
        for marker in forbidden_path_markers
    )
    split_path = (
        "infer_cli.py" in string_literals
        and "f5_tts" in string_literals
        and "infer" in string_literals
    )
    direct_import = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            direct_import = any(alias.name == "f5_tts.infer.infer_cli" for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            direct_import = module == "f5_tts.infer.infer_cli" or (
                module == "f5_tts.infer"
                and any(alias.name == "infer_cli" for alias in node.names)
            )
        if direct_import:
            break
    if direct_path or split_path or direct_import:
        errors.append(
            "TTS candidate calls raw F5-TTS infer_cli.py; "
            "use SURE_TTS_BATCH_INFER_WRAPPER so the model and vocoder are loaded once per worker"
        )
    return errors


def _validate_asr_subprocess_safety(tree: ast.AST) -> list[str]:
    errors: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func_name = _call_name(node.func)
        if func_name not in {"subprocess.Popen", "Popen", "subprocess.run", "run"}:
            continue

        for keyword in node.keywords:
            if keyword.arg == "capture_output" and _is_truthy_ast_constant(keyword.value):
                errors.append(
                    "ASR candidate uses subprocess capture_output=True; "
                    "write long-running train/decode output directly to a log file instead"
                )
            elif keyword.arg in {"stdout", "stderr"} and _is_subprocess_pipe(keyword.value):
                errors.append(
                    f"ASR candidate uses {keyword.arg}=subprocess.PIPE; "
                    "long-running icefall/DDP commands must write directly to log files "
                    "or use a process-group-safe helper"
                )
    return errors


def _validate_asr_cli_compatibility(tree: ast.AST) -> list[str]:
    errors: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.List, ast.Tuple)):
            continue
        command_literals = _collect_string_literals(node)
        targets_decode = _command_targets_script(command_literals, "decode.py") or (
            _command_targets_script(command_literals, "decode_sure_eval.py")
        )
        if not targets_decode:
            continue
        bad_args = sorted(
            set(command_literals).intersection(ASR_ZIPFORMER_DECODE_TRAIN_ONLY_ARGS)
        )
        if bad_args:
            errors.append(
                "passes Zipformer training-only argument(s) to decode.py: "
                f"{', '.join(bad_args)}; pass loss/SpecAugment options only to train.py"
            )
    return errors


def _validate_asr_lhotse_key_normalization(code_content: str) -> list[str]:
    bad_suffix_length_checks = (
        r"len\s*\(\s*parts\s*\[\s*-1\s*\]\s*\)\s*(?:<=|<|==)\s*\d+",
        r"len\s*\(\s*parts\s*\[\s*len\s*\(\s*parts\s*\)\s*-\s*1\s*\]\s*\)"
        r"\s*(?:<=|<|==)\s*\d+",
        r"len\s*\(\s*(?:suffix|segment_suffix|cut_suffix|lhotse_suffix)\s*\)"
        r"\s*(?:<=|<|==)\s*\d+",
    )
    if any(re.search(pattern, code_content) for pattern in bad_suffix_length_checks):
        return [
            "ASR candidate restricts Lhotse numeric suffix length; when the active "
            "dataset profile uses numeric-suffix normalization, remove any trailing "
            "numeric suffix regardless of length before writing artifacts/hyp.txt"
        ]
    return []


def _validate_asr_duration_resolution_safety(code_content: str) -> list[str]:
    errors: list[str] = []
    risky_markers = (
        "parse_any_integer_from_log",
        "failed but log contains usable max-duration",
        "helper attempt",
        "duration_autotune_fallback.log",
    )
    if "resolve-max-duration" not in code_content and "SURE_RUNTIME_ENV_HELPER" not in code_content:
        return errors
    if any(marker in code_content for marker in risky_markers):
        errors.append(
            "ASR candidate attempts to recover SURE_MAX_DURATION from failed helper "
            "logs; only accept the final standalone integer line when "
            "SURE_RUNTIME_ENV_HELPER resolve-max-duration exits successfully"
        )

    failed_log_regex = re.compile(
        r"re\.search\([^)]*(?:resolved|max\[-_ \]duration|selected|final)[^)]*",
        re.DOTALL,
    )
    if failed_log_regex.search(code_content) and "except" in code_content:
        errors.append(
            "ASR candidate uses regex-based max-duration recovery in an exception "
            "path; do not parse tracebacks, command lines, or failed helper logs for "
            "duration values"
        )

    try:
        tree = ast.parse(code_content)
    except SyntaxError:
        return errors
    errors.extend(_validate_asr_duration_helper_invocation(tree))
    return errors


def _validate_asr_duration_helper_invocation(tree: ast.AST) -> list[str]:
    errors: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func_name = _call_name(node.func)
        if func_name not in {
            "subprocess.Popen",
            "Popen",
            "subprocess.run",
            "run",
            "subprocess.check_call",
            "check_call",
            "subprocess.check_output",
            "check_output",
        }:
            continue
        if not node.args:
            continue
        sequence = _literal_command_sequence(node.args[0])
        if "resolve-max-duration" not in sequence:
            continue
        resolve_index = sequence.index("resolve-max-duration")
        prefix = sequence[:resolve_index]
        uses_python = any(_looks_like_python_launcher(part) for part in prefix)
        if not uses_python:
            errors.append(
                "ASR candidate invokes SURE_RUNTIME_ENV_HELPER directly for resolve-max-duration; "
                "run it through a Python executable such as sys.executable or "
                "[os.environ.get('SURE_ICEFALL_PYTHON', sys.executable), helper, 'resolve-max-duration']"
            )
    return errors


def _validate_asr_zipformer_wrapper_boundary(
    tree: ast.AST,
    code_content: str,
    *,
    require_asr_wrapper: bool,
) -> list[str]:
    if not require_asr_wrapper:
        return []
    if "official_baseline.json" in code_content and "zipformer_large_cr_ctc_rnnt" in code_content:
        return []
    if any(
        marker in code_content
        for marker in (
            "SURE_RUNTIME_ENV_HELPER",
            "resolve-max-duration",
        )
    ):
        return [
            "ASR Zipformer candidate must not call the duration helper directly; "
            "pass candidate train/decode args through SURE_ASR_ZIPFORMER_WRAPPER "
            "so duration probing, environment setup, checkpoint validation, decode "
            "split handling, and hyp formatting stay inside the SURE Master contract"
        ]
    if "SURE_ASR_ZIPFORMER_WRAPPER" in code_content or "run_icefall_zipformer_candidate.py" in code_content:
        return []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.List, ast.Tuple)):
            continue
        command_literals = _collect_string_literals(node)
        if _command_targets_script(command_literals, "train.py") or _command_targets_script(command_literals, "decode.py"):
            return [
                "ASR Zipformer candidate calls train.py/decode.py directly; use "
                "SURE_ASR_ZIPFORMER_WRAPPER so training, duration probing, checkpoint "
                "validation, decode split handling, and hyp formatting stay inside "
                "the SURE Master contract"
            ]
    return []


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parent = _call_name(func.value)
        return f"{parent}.{func.attr}" if parent else func.attr
    return ""


def _is_truthy_ast_constant(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and bool(node.value)


def _is_subprocess_pipe(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "PIPE"
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "PIPE"
        and isinstance(node.value, ast.Name)
        and node.value.id == "subprocess"
    )


def _literal_command_sequence(node: ast.AST) -> list[str]:
    if not isinstance(node, (ast.List, ast.Tuple)):
        return []
    result: list[str] = []
    for item in node.elts:
        if isinstance(item, ast.Constant) and isinstance(item.value, str):
            result.append(item.value)
        elif isinstance(item, ast.Attribute):
            result.append(_call_name(item))
        elif isinstance(item, ast.Name):
            result.append(item.id)
        elif isinstance(item, ast.Call):
            result.append(_call_name(item.func))
        elif isinstance(item, ast.Subscript):
            result.append(_call_name(item.value))
    return result


def _looks_like_python_launcher(value: str) -> bool:
    text = value.strip()
    base = Path(text).name.lower()
    return (
        text == "sys.executable"
        or "SURE_ICEFALL_PYTHON" in text
        or base in {"python", "python3"}
        or base.startswith("python3.")
        or base.startswith("python-")
    )


def _validate_base_model_boundary(
    code_content: str,
    tree: ast.AST,
    profile: BaseModelProfile,
) -> list[str]:
    errors: list[str] = []
    string_literals = _collect_string_literals(tree)
    target_paths = [
        str(path).strip()
        for path in profile.required_paths.values()
        if str(path).strip()
    ]
    if profile.is_required and target_paths:
        if not any(
            _references_workspace_path(path, code_content, string_literals)
            for path in target_paths
        ):
            joined = ", ".join(target_paths)
            errors.append(
                "required base model profile is not referenced; "
                f"use workspace-relative path(s): {joined}"
            )

    source_roots: set[str] = set()
    for source in profile.source_paths.values():
        if not source:
            continue
        source_path = Path(str(source))
        source_roots.add(str(source_path))
        try:
            source_roots.add(str(source_path.resolve()))
        except OSError:
            pass
    for source_root in source_roots:
        if source_root and source_root in code_content:
            errors.append(
                "references base model source path directly; "
                f"use workspace-relative symlink instead: {source_root}"
            )

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_os_chdir_call(node):
            first_arg = node.args[0] if node.args else None
            if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                if any(first_arg.value == source for source in source_roots):
                    errors.append(
                        "changes directory to base model source path directly; "
                        "use workspace-relative symlink instead"
                    )
        elif isinstance(node, (ast.List, ast.Tuple)):
            command_literals = _collect_string_literals(node)
            if _command_targets_script(command_literals, "train.py") and "--lang-dir" in command_literals:
                errors.append(
                    "passes --lang-dir to zipformer train.py; "
                    "use --bpe-model for training and reserve --lang-dir for decode.py"
                )
    return errors


def _collect_string_literals(tree: ast.AST) -> set[str]:
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def _references_workspace_path(
    path: str,
    code_content: str,
    string_literals: set[str],
) -> bool:
    normalized = path.replace("\\", "/").strip()
    if not normalized:
        return False
    if normalized in code_content or normalized in string_literals:
        return True

    parts = [part for part in Path(normalized).parts if part not in ("", ".")]
    return len(parts) > 1 and all(part in string_literals for part in parts)


def _command_targets_script(string_literals: set[str], script_name: str) -> bool:
    return any(
        value == script_name
        or value.endswith("/" + script_name)
        or value.endswith("\\" + script_name)
        for value in string_literals
    )


def _is_os_chdir_call(node: ast.Call) -> bool:
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "chdir"
        and isinstance(func.value, ast.Name)
        and func.value.id == "os"
    )
