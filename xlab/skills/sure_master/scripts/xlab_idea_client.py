"""JSONL adapter between SURE contracts and package-native XLab research ideas."""
from __future__ import annotations

import hashlib
import contextlib
import json
import os
import sys
import tempfile
import shlex
from pathlib import Path
from typing import Any, Mapping

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "research_idea" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from research_idea_lib.common import read_json, run_paths  # noqa: E402
from research_idea_lib.config import load_runtime_config  # noqa: E402
from research_idea_lib.survey_repository import SurveyArtifactRepository
from research_idea_lib.inputs import IdeaRequest as NativeIdeaRequest, parse_request_args  # noqa: E402
from research_idea_lib.pipeline import run_pipeline  # noqa: E402
from research_idea_lib.research_idea_artifacts import final_idea_result  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sure_idea_review import review_candidate, scientific_fingerprint, validate_review  # noqa: E402

PROTOCOL = "xlab.sure.jsonl.v1"
SUMMARY_SCHEMA = "sure.round_summary.v1"
_NATIVE_REQUIRED_TEXT = (
    "title",
    "abstract",
    "core_contribution",
    "research_question",
    "hypothesis",
    "method",
    "introduction",
)
_NATIVE_REQUIRED_LISTS = (
    "experiment_plan",
    "data_requirements",
    "baselines",
    "metrics",
    "risks",
    "components",
    "algorithm",
    "reference_papers",
)
_CANDIDATE_TYPES = {"arch", "fine_tune", "inference"}


class AdapterError(RuntimeError):
    """Raised when an operation cannot produce a valid typed response."""


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AdapterError(f"generated idea field {field} is empty")
    return value.strip()


def _safe_run_dir(payload: Mapping[str, Any], operation_id: str) -> tuple[Path, Path]:
    root = Path(str(payload.get("run_root") or os.environ.get("XLAB_SURE_RUN_ROOT")
                    or Path(tempfile.gettempdir()) / "xlab-sure-runs"))
    root = root.expanduser().resolve()
    requested = str(payload.get("run_dir") or operation_id)
    candidate = Path(requested)
    if candidate.is_absolute():
        run_dir = candidate.resolve()
        try:
            run_dir.relative_to(root)
        except ValueError as exc:
            raise AdapterError("run_dir must be inside run_root") from exc
    else:
        if ".." in candidate.parts:
            raise AdapterError("run_dir must not escape run_root")
        run_dir = (root / candidate).resolve()
    return root, run_dir


def _survey_path(payload: Mapping[str, Any], root: Path) -> Path:
    raw = str(payload.get("survey_path") or os.environ.get("XLAB_SURE_SURVEY_PATH") or "")
    if not raw:
        raise AdapterError("generate requires survey_path or XLAB_SURE_SURVEY_PATH")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (root / path).resolve()
    else:
        path = path.resolve()
    if not path.exists():
        raise AdapterError("configured survey_path does not exist")
    return path


def _native_arguments(
    payload: Mapping[str, Any],
    survey: Path,
    feedback: str = "",
    *,
    candidate_index: int | None = None,
    accepted: list[dict[str, Any]] | None = None,
    rejected: list[dict[str, Any]] | None = None,
) -> str:
    task = _required_text(payload.get("task_description"), "task_description")
    axis = str(payload.get("axis") or "")
    mode = f"Generate a SURE {axis} candidate." if axis else "Generate a SURE research candidate."
    mode += (
        " Freely choose the most valuable evidence-backed hypothesis within the execution contract. "
        "There are no architecture/training/inference quotas. One hypothesis may require coordinated "
        "changes across these domains; specify its ablation. Different wording or another numeric "
        "setting alone does not make a distinct candidate. Account for accepted and rejected attempts."
    )
    discussion = canonical({
        "search_mode": payload.get("search_mode"),
        "axis": payload.get("axis"),
        "axis_index": payload.get("axis_index"),
        "phase": payload.get("phase", "research"),
        "round_index": payload.get("round_index"),
        "current_best": payload.get("current_best", {}),
        "task_card": payload.get("task_card", {}),
        "base_model_profile": payload.get("base_model_profile", {}),
        "metric": payload.get("metric", {}),
        "execution_contract": payload.get("execution_contract", {}),
        "history": payload.get("prior_rounds", payload.get("history", [])),
        "history_digest": payload.get("history_digest"),
        "parent_lineage": payload.get("parent_lineage", payload.get("history_artifacts", [])),
        "candidate_index": candidate_index,
        "accepted_candidates": accepted or [],
        "rejected_attempts": rejected or [],
    })
    parts = [
        "--survey", str(survey),
        "--topic", task,
        "--discussion", f"{mode} Preserve typed implementation details. Context: {discussion}",
    ]
    parent = payload.get("current_best", {}).get("native_idea")
    if feedback and isinstance(parent, dict) and parent.get("title") and parent.get("components"):
        if payload.get("current_best", {}).get("native_idea_digest") != digest(parent):
            raise AdapterError("mature idea digest mismatch")
        parts.extend(["--mature-idea", canonical(parent), "--experiment-feedback", feedback])
    return shlex.join(parts)


def _native_candidate_payload(artifact: Mapping[str, Any]) -> dict[str, Any]:
    nested = final_idea_result(artifact)
    result = nested if nested else artifact
    if not isinstance(result, Mapping) or not result:
        raise AdapterError("package-native research idea generation did not return a fused idea")
    candidate = dict(result)
    if not all(isinstance(candidate.get(field), str) and candidate[field].strip() for field in _NATIVE_REQUIRED_TEXT):
        raise AdapterError(
            "package-native research idea generation returned one fused idea; "
            "SURE requires four distinct ideas and the adapter will not clone it"
        )
    return candidate


def _native_candidate_type(payload: Mapping[str, Any], candidate: Mapping[str, Any]) -> str:
    axis = str(payload.get("axis") or "")
    expected = {"arch": "arch", "train": "fine_tune", "inference": "inference"}.get(axis)
    value = str(candidate.get("candidate_type") or expected or "")
    if value not in _CANDIDATE_TYPES:
        raise AdapterError(f"native candidate has unsupported candidate_type: {value}")
    if expected and value != expected:
        raise AdapterError("native candidate candidate_type does not match requested axis")
    return value


def _native_candidate_fingerprint(payload: Mapping[str, Any], candidate: Mapping[str, Any]) -> str:
    return scientific_fingerprint(candidate)


def _validate_native_candidate(payload: Mapping[str, Any], candidate: Mapping[str, Any]) -> None:
    for field in _NATIVE_REQUIRED_TEXT:
        _required_text(candidate.get(field), field)
    for field in _NATIVE_REQUIRED_LISTS:
        value = candidate.get(field)
        if not isinstance(value, list) or not value:
            raise AdapterError(f"native candidate field {field} must be a non-empty list")
    if not isinstance(candidate.get("evidence_ids"), list) or not candidate["evidence_ids"]:
        raise AdapterError("native candidate evidence_ids must be a non-empty list")
    if any(not isinstance(item, str) or not item.strip() for item in candidate["evidence_ids"]):
        raise AdapterError("native candidate evidence_ids must contain non-empty strings")


def _portable_reference(value: str, field: str) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or str(path) in {"", "."}:
        raise AdapterError(f"{field} must be a portable workspace-relative path")
    return path.as_posix()


def _validate_batch_payload(batch: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    required = {"schema_version", "status", "request_digest", "xlab_run_id", "workspace", "ideas", "batch_digest"}
    if not required.issubset(batch):
        raise AdapterError("sure.idea_batch.v1 response is missing required fields")
    if batch.get("schema_version") != "sure.idea_batch.v1" or batch.get("status") != "success":
        raise AdapterError("sure.idea_batch.v1 response has invalid schema or status")
    if batch.get("request_digest") != str(payload.get("input_digest") or digest(payload)):
        raise AdapterError("sure.idea_batch.v1 request digest mismatch")
    workspace = _portable_reference(str(batch.get("workspace") or ""), "workspace")
    ideas = batch.get("ideas")
    if not isinstance(ideas, list) or len(ideas) != 4 or any(not isinstance(item, Mapping) for item in ideas):
        raise AdapterError("sure.idea_batch.v1 must contain exactly four idea objects")
    expected_type = {"arch": "arch", "train": "fine_tune", "inference": "inference"}.get(str(payload.get("axis") or ""))
    idea_ids: set[str] = set()
    artifact_ids: set[str] = set()
    artifact_digests: set[str] = set()
    fingerprints: set[str] = set()
    for idea in ideas:
        idea_id = _required_text(idea.get("idea_id"), "idea_id")
        artifact_id = _portable_reference(_required_text(idea.get("artifact_id"), "artifact_id"), "artifact_id")
        artifact_digest = _required_text(idea.get("artifact_digest"), "artifact_digest")
        if not artifact_digest.startswith("sha256:"):
            raise AdapterError("artifact_digest must use sha256")
        if idea_id in idea_ids or artifact_id in artifact_ids or artifact_digest in artifact_digests:
            raise AdapterError("sure.idea_batch.v1 idea lineage values must be unique")
        idea_ids.add(idea_id)
        artifact_ids.add(artifact_id)
        artifact_digests.add(artifact_digest)
        if expected_type and (idea.get("axis") != payload.get("axis") or idea.get("candidate_type") != expected_type):
            raise AdapterError("sure.idea_batch.v1 axis or candidate_type mismatch")
        spec = idea.get("spec")
        if not isinstance(spec, Mapping) or not str(spec.get("implementation_instructions") or "").strip():
            raise AdapterError("sure.idea_batch.v1 idea spec is incomplete")
        fingerprint = digest({
            "hypothesis": str(idea.get("hypothesis") or "").strip(),
            "mechanism": str(idea.get("mechanism") or "").strip(),
            "change_set": spec.get("change_set"),
        })
        if fingerprint in fingerprints:
            raise AdapterError("sure.idea_batch.v1 contains duplicate scientific candidates")
        fingerprints.add(fingerprint)
    expected_batch_digest = digest({key: value for key, value in batch.items() if key != "batch_digest"})
    if batch.get("batch_digest") != expected_batch_digest:
        raise AdapterError("sure.idea_batch.v1 batch digest mismatch")
    del workspace


def _idea_batch_from_candidates(
    payload: Mapping[str, Any],
    operation_id: str,
    candidates: list[dict[str, Any]],
    root: Path,
    run_dir: Path,
) -> dict[str, Any]:
    if len(candidates) != 4:
        raise AdapterError("SURE native generation must produce exactly four candidates")

    request_digest = str(payload.get("input_digest") or digest(payload))
    axis = payload.get("axis")
    expected_type = {"arch": "arch", "train": "fine_tune", "inference": "inference"}.get(str(axis))
    ideas: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    artifact_digests: set[str] = set()
    for index, record in enumerate(candidates, start=1):
        candidate = record["native"]
        review = validate_review(record["review"], axis)
        _validate_native_candidate(payload, candidate)
        candidate_type = review["candidate_type"]
        fingerprint = _native_candidate_fingerprint(payload, {**candidate, "candidate_type": candidate_type})
        if fingerprint in fingerprints:
            raise AdapterError("native candidates are not scientifically distinct")
        fingerprints.add(fingerprint)
        artifact_id = record["artifact_id"]
        artifact_digest = record["artifact_digest"]
        if artifact_digest in artifact_digests:
            raise AdapterError("native candidate artifact digests must be unique")
        artifact_digests.add(artifact_digest)
        spec = {
            "implementation_instructions": review["implementation_instructions"],
            "change_set": review["change_set"],
            "change_domains": review["change_domains"],
            "requires_training": review["requires_training"],
            "invariants": ["preserve evaluator and artifact contract"],
            "success_criteria": [str(item) for item in candidate["metrics"] if isinstance(item, str) and item.strip()],
            "ablation": review["ablation"],
            "expected_effect": str(candidate["core_contribution"]).strip(),
            "risks": [str(item) for item in candidate["risks"] if isinstance(item, str) and item.strip()],
            "resource_class": "native-xlab",
        }
        ideas.append({
            "idea_id": f"{operation_id}-candidate-{index}",
            "artifact_id": artifact_id,
            "artifact_digest": artifact_digest,
            "title": str(candidate["title"]).strip(),
            "axis": axis if axis in {"arch", "train", "inference"} else None,
            "candidate_type": candidate_type,
            "hypothesis": str(candidate["hypothesis"]).strip(),
            "mechanism": str(candidate.get("mechanism") or candidate["method"]).strip(),
            "evidence_refs": [str(item) for item in candidate["evidence_ids"]],
            "novelty": {"native_fingerprint": fingerprint, "source_modes": candidate.get("source_modes", []),
                        "distinction": review["distinction"], "native_artifact_id": record["native_artifact_id"]},
            "spec": spec,
            "native_artifact": candidate,
        })

    if expected_type and any(item["candidate_type"] != expected_type for item in ideas):
        raise AdapterError("native candidate candidate_type does not match requested axis")
    try:
        workspace = run_dir.relative_to(root).as_posix()
    except ValueError as exc:
        raise AdapterError("native candidate run directory escaped run_root") from exc
    batch = {
        "schema_version": "sure.idea_batch.v1",
        "status": "success",
        "request_digest": request_digest,
        "xlab_run_id": operation_id,
        "workspace": workspace,
        "ideas": ideas,
    }
    batch["batch_digest"] = digest(batch)
    # Keep the run directory in the API for lineage checks without exposing an absolute path.
    if not run_dir.is_dir():
        raise AdapterError("native candidate run directory is unavailable")
    _validate_batch_payload(batch, payload)
    return batch


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_survey_resources(survey: Path) -> dict[str, Any]:
    repository = SurveyArtifactRepository.from_request(
        NativeIdeaRequest(survey_path=survey), Path(__file__).resolve().parents[4])
    validation = repository.validation()
    blockers = validation["blocking_errors"] + repository.source.resources.blocking_errors
    if blockers:
        raise AdapterError("; ".join(blockers))
    return {"status": "ready", "survey_validated": True, "resources_validated": True}


def generate(payload: Mapping[str, Any], operation_id: str) -> dict[str, Any]:
    if payload.get("requested_idea_count") != 4:
        raise AdapterError("SURE generate requires exactly four ideas")
    max_attempts = payload.get("generation_policy", {}).get("max_attempts", 8)
    if type(max_attempts) is not int or max_attempts < 4:
        raise AdapterError("generation max_attempts must be an integer >= 4")
    root, run_dir = _safe_run_dir(payload, operation_id)
    request_digest = digest(payload)
    state_path = run_dir / "generation.json"
    if state_path.exists():
        previous = read_json(state_path)
        if previous.get("request_digest") != request_digest:
            raise AdapterError("operation identity conflicts with existing request")
        if previous.get("status") == "published":
            batch = read_json(run_dir / "batch.json")
            _validate_batch_payload(batch, payload)
            return batch
        raise AdapterError("generation is incomplete; inspect existing attempts before retrying")
    survey = _survey_path(payload, root)
    validate_survey_resources(survey)
    runtime = load_runtime_config()
    native_cwd = Path(__file__).resolve().parents[4]
    state = {"request_digest": request_digest, "status": "started", "attempts": [], "accepted": []}
    _write_json(state_path, state)
    candidates: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    history = payload.get("prior_rounds", [])
    feedback = canonical({"rounds": history}) if history else ""
    try:
        for attempt in range(1, max_attempts + 1):
            branch_id = f"{operation_id}-attempt-{attempt}"
            branch_dir = run_dir / f"attempt-{attempt}"
            args = _native_arguments(payload, survey, feedback, candidate_index=attempt - 1,
                                     accepted=accepted, rejected=rejected)
            native_request = parse_request_args(args, native_cwd, runtime)
            attempt_record = {"attempt": attempt, "status": "started", "input_digest": digest(args)}
            state["attempts"].append(attempt_record)
            _write_json(branch_dir / "sure_request.json", {"args": args, "input_digest": digest(args)})
            _write_json(state_path, state)
            result = run_pipeline(cwd=native_cwd, run_dir=branch_dir, run_id=branch_id,
                                  request=native_request, runtime=runtime)
            if result.get("status") != "success":
                raise AdapterError(f"package-native attempt {attempt} was incomplete")
            artifact = read_json(run_paths(branch_dir)["idea_result_json"])
            native = {}
            try:
                native = _native_candidate_payload(artifact)
                _validate_native_candidate(payload, native)
                fingerprint = scientific_fingerprint(native)
                if fingerprint in fingerprints:
                    raise ValueError("duplicate scientific candidate")
                context = {"candidate": native, "accepted_candidates": accepted,
                           "rejected_attempts": rejected, "prior_rounds": history,
                           "current_best": payload.get("current_best", {}),
                           "execution_contract": payload.get("execution_contract", {}),
                           "task_card": payload.get("task_card", {}),
                           "base_model_profile": payload.get("base_model_profile", {}),
                           "axis": payload.get("axis")}
                response = review_candidate(runtime, context)
                _write_json(branch_dir / "review.json", response)
                review = validate_review(response.get("review"), payload.get("axis"))
            except (ValueError, AdapterError) as exc:
                rejection = {"attempt": attempt, "candidate": native, "reason": str(exc)}
                rejected.append(rejection)
                attempt_record.update(status="rejected", reason=str(exc))
                _write_json(state_path, state)
                continue
            fingerprints.add(fingerprint)
            accepted_document = {"native": native, "review": review,
                                 "native_artifact_id": str(run_paths(branch_dir)["idea_result_json"].relative_to(root))}
            artifact_path = branch_dir / "accepted.json"
            _write_json(artifact_path, accepted_document)
            record = {**accepted_document, "artifact_id": str(artifact_path.relative_to(root)),
                      "artifact_digest": digest(accepted_document)}
            candidates.append(record)
            accepted.append({"hypothesis": native["hypothesis"], "mechanism": native.get("mechanism", native["method"]),
                             "change_set": review["change_set"], "fingerprint": fingerprint})
            attempt_record.update(status="accepted", artifact_id=record["artifact_id"])
            state["accepted"].append(record["artifact_id"])
            _write_json(state_path, state)
            if len(candidates) == 4:
                batch = _idea_batch_from_candidates(payload, operation_id, candidates, root, run_dir)
                _write_json(run_dir / "batch.json", batch)
                state["status"] = "published"
                _write_json(state_path, state)
                return batch
        raise AdapterError(f"generation incomplete: accepted {len(candidates)} of 4 after {max_attempts} attempts")
    except Exception as exc:
        state.update(status="incomplete", error_type=type(exc).__name__)
        _write_json(state_path, state)
        raise


def summarize(payload: Mapping[str, Any], operation_id: str) -> dict[str, Any]:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise AdapterError("summarize requires candidates")
    failure_counts: dict[str, int] = {}
    attempted: list[str] = []
    supported: list[str] = []
    rejected: list[str] = []
    observations: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        idea_id = str(candidate.get("idea_id") or "")
        category = candidate.get("failure_category")
        idea = candidate.get("idea") if isinstance(candidate.get("idea"), dict) else {}
        mechanism = str(idea.get("mechanism") or idea.get("title") or idea_id)
        fingerprint = digest({"hypothesis": idea.get("hypothesis"), "mechanism": idea.get("mechanism"),
                              "change_set": idea.get("spec", {}).get("change_set", [])})
        attempted.append(fingerprint)
        if category:
            failure_counts[str(category)] = failure_counts.get(str(category), 0) + 1
        measured = any(rung.get("success") and isinstance(rung.get("score"), (int, float))
                       and not isinstance(rung.get("score"), bool) for rung in candidate.get("rungs", []))
        if category or candidate.get("final_status") != "success":
            outcome = "execution_failed"
        elif not measured:
            outcome = "inconclusive"
        elif candidate.get("improved"):
            outcome = "improved"
            supported.append(mechanism)
        else:
            outcome = "no_improvement"
        observations.append({"idea_id": idea_id, "mechanism": mechanism, "outcome": outcome,
                             "failure_category": category, "reason_code": candidate.get("reason_code"),
                             "rungs": candidate.get("rungs", []),
                             "interpretation": "Observation under the recorded budget and evaluation scope; not a general mechanism verdict."})
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "sure_run_id": str(payload.get("sure_run_id") or ""),
        "search_mode": payload.get("search_mode"),
        "round_index": int(payload.get("round_index") or 0),
        "summary_digest": digest({"operation_id": operation_id, "result_digest": payload.get("result_digest"), "failure_counts": failure_counts}),
        "observations": observations,
        "supported_mechanisms": supported,
        "rejected_mechanisms": rejected,
        "failure_counts": failure_counts,
        "next_directions": ["Use measured outcomes and their budgets to choose the next hypotheses. Resolve execution failures before drawing scientific conclusions; justify revisits with new conditions or unresolved questions."],
        "attempted_idea_fingerprints": attempted,
        "source_artifacts": [str(item) for item in payload.get("artifact_refs", []) if isinstance(item, str)],
        "history_digest": str(payload.get("history_digest") or ""),
        "feedback_digest": str(payload.get("feedback_digest") or payload.get("result_digest") or ""),
        "parent_artifacts": [str(item) for item in payload.get("parent_artifacts", []) if isinstance(item, str)],
        "phase": payload.get("phase", "research"),
        "axis": payload.get("axis"),
    }
    summary["summary_digest"] = digest({key: value for key, value in summary.items() if key != "summary_digest"})
    return summary


def dispatch(envelope: Mapping[str, Any]) -> dict[str, Any]:
    if envelope.get("protocol") != PROTOCOL:
        raise AdapterError("unsupported protocol")
    operation = envelope.get("operation")
    operation_id = envelope.get("operation_id")
    payload = envelope.get("payload")
    if not isinstance(operation_id, str) or not operation_id.strip():
        raise AdapterError("operation_id must be non-empty")
    if not isinstance(payload, dict):
        raise AdapterError("payload must be an object")
    if operation == "generate":
        value = generate(payload, operation_id)
    elif operation == "summarize":
        value = summarize(payload, operation_id)
    else:
        raise AdapterError("unsupported operation")
    return {"protocol": PROTOCOL, "operation": operation, "operation_id": operation_id, "status": "success", "payload": value}


def main() -> int:
    envelope = None
    if len(sys.argv) == 3 and sys.argv[1] == "--check-survey":
        try:
            result = validate_survey_resources(Path(sys.argv[2]))
        except Exception as exc:
            print(json.dumps({"status": "blocked", "error": str(exc)}))
            return 2
        print(json.dumps(result))
        return 0
    line = sys.stdin.readline()
    try:
        envelope = json.loads(line)
        with contextlib.redirect_stdout(sys.stderr):
            response = dispatch(envelope)
        print(json.dumps(response, ensure_ascii=False, sort_keys=True), flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001 - protocol must return a machine-readable failure.
        operation_id = envelope.get("operation_id") if isinstance(envelope, dict) else "unknown"
        print(json.dumps({"protocol": PROTOCOL, "operation_id": operation_id, "status": "failed", "error": str(exc)}, ensure_ascii=False), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
