#!/usr/bin/env python3
"""Create publication.json and publication_artifacts.json for model that passed onboarding."""
import hashlib, json, sys
from pathlib import Path
from datetime import datetime, timezone

MODEL_DIR = Path("/hpc_stor03/project/oref/nfs/models/openai__whisper-large-v3-turbo")
MODEL_ID = "openai__whisper-large-v3-turbo"
EXCLUDED_PARTS = {".runtime", "__pycache__", "eval_runs", ".git", "eval_runs_514", "eval_runs_v0", "fixture", "docker_artifacts", "repro_bundle", ".venv", ".venv.hostbak"}

def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

files = []
for path in sorted(MODEL_DIR.rglob("*")):
    relative = path.relative_to(MODEL_DIR)
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        continue
    if relative.name in {"publication.json", "publication_artifacts.json"}:
        continue
    if path.is_symlink():
        continue
    if path.is_file():
        files.append({
            "path": relative.as_posix(),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        })

artifact_manifest = {
    "schema": "sure.eval.model_artifact_manifest.v1",
    "model_id": MODEL_ID,
    "files": files,
}

manifest_path = MODEL_DIR / "publication_artifacts.json"
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(artifact_manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
    f.write("\n")

aggregate_sha256 = hashlib.sha256(
    json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()

publication = {
    "schema": "sure.eval.model_publication.v1",
    "status": "verified",
    "model_id": MODEL_ID,
    "artifact_sha256": aggregate_sha256,
    "artifact_manifest_path": "publication_artifacts.json",
    "artifact_manifest_sha256": file_sha256(manifest_path),
    "verified_by": "sure-eval-onboard-system",
    "verified_at": utc_now(),
}

pub_path = MODEL_DIR / "publication.json"
with open(pub_path, "w", encoding="utf-8") as f:
    json.dump(publication, f, ensure_ascii=False, indent=2, sort_keys=True)
    f.write("\n")

print(f"Created publication.json (artifact_sha256={aggregate_sha256})")
print(f"Created publication_artifacts.json ({len(files)} files)")
print(f"Files indexed:")
for f in files:
    print(f"  {f['path']} sha256={f['sha256'][:16]}... size={f['size_bytes']}")
