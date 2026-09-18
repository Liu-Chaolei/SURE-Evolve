"""Named SD deployments, verified archives and a shared execution lease."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from playground.sure_master.core.artifacts import file_digest
from playground.sure_master.core.utils.slurm import atomic_json
from playground.sure_master.core.utils.slurm_allocations import process_identity
from playground.sure_master.core.utils.fingerprints import digest

RUNS = Path(__file__).resolve().parents[3] / "runs"


def live_processes(root: Path) -> list[int]:
    result = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            args = (process / "cmdline").read_bytes().decode().split("\0")
            belongs = any(arg == str(root) or arg.startswith(str(root) + "/") for arg in args)
            if process.stat().st_uid == os.getuid() and belongs:
                # Exclude the management wrapper itself; only experiment processes matter.
                if "playground.sure_master.tools.sd_deployments" not in args and process_identity(int(process.name)):
                    result.append(int(process.name))
        except (OSError, UnicodeError):
            continue
    return result


def inventory(root: Path) -> dict:
    files, links = {}, {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.name == ".env" or path.name.startswith(".env."):
            continue
        if path.is_symlink():
            links[relative] = os.readlink(path)
        elif path.is_file():
            files[relative] = {"sha256": file_digest(path), "bytes": path.stat().st_size}
    return {"files": files, "links": links}


def register(runs: Path, name: str, record: dict) -> None:
    runs.mkdir(parents=True, exist_ok=True)
    with (runs / "sd_deployments.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = runs / "sd_deployments.json"
        document = json.loads(path.read_text()) if path.exists() else {"schema": "sure.sd_deployments.v1", "deployments": {}}
        previous = document["deployments"].get(name)
        if previous and previous["root"] != record["root"]:
            raise ValueError("Deployment name is already bound to another directory")
        document["deployments"][name] = {**(previous or {}), **record}
        atomic_json(path, document)


def archive(root: Path, name: str, runs: Path) -> dict:
    if live_processes(root):
        raise RuntimeError("Wait for current operations to stop before archiving")
    destination = runs / "archives" / name
    if destination.exists():
        raise ValueError("Archive already exists; do not overwrite it")
    metadata = {"name": name, "root": str(root), "status": "archived_incomplete", "archived_at": time.time(),
                "resume_policy": "Keep original absolute paths; reconcile incomplete provider operations before retry."}
    atomic_json(root / "deployment_name.json", metadata)
    environment = {}
    for role, python in {"controller": "/shared/chaolei.liu/data/sure_asr_controller/bin/python",
                         "research": "/shared/chaolei.liu/data/sure_xlab_runtime312/bin/python"}.items():
        environment[role] = {"python": python,
            "version": subprocess.check_output([python, "--version"], text=True).strip(),
            "packages": subprocess.check_output([python, "-m", "pip", "freeze"], text=True).splitlines()}
    atomic_json(root / "archive_environment.json", environment)
    source_manifest = inventory(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(root, destination, symlinks=True, ignore=shutil.ignore_patterns(".env", ".env.*"))
    if inventory(destination) != source_manifest:
        raise RuntimeError("Archive content verification failed")
    manifest = runs / "archives" / (name + ".manifest.json")
    atomic_json(manifest, {"root": str(root), "backup": str(destination), **source_manifest})
    record = {**metadata, "backup": str(destination), "manifest": str(manifest),
              "files_verified": len(source_manifest["files"])}
    register(runs, name, record)
    return record


def check_archive(record: dict) -> dict:
    manifest = json.loads(Path(record["manifest"]).read_text())
    actual = inventory(Path(record["backup"]))
    if actual != {key: manifest[key] for key in ("files", "links")}:
        raise ValueError("Archive checksum or link verification failed")
    dependencies = record.get("external_dependencies")
    if dependencies:
        for name, entry in json.loads(Path(dependencies).read_text())["files"].items():
            path = Path(name)
            if not path.is_file() or file_digest(path) != entry["sha256"]:
                raise ValueError(f"Archived shared dependency changed: {path}")
    return {"status": "verified", "name": record["name"], "files": len(actual["files"])}


def retry_failed(root: Path, group: str, operation: str) -> None:
    """Explicit recovery of an inspected failed API operation; never sends a request."""
    if live_processes(root):
        raise RuntimeError("Stop the deployment before reconciling its provider receipts")
    path = root / group / "search/workspace/artifacts/xlab_operations.json"
    document = json.loads(path.read_text())
    base = {key: document[key] for key in ("schema_version", "operations")}
    if digest(base) != document.get("document_digest"):
        raise ValueError("Provider receipt checksum mismatch")
    entry = document["operations"].get(operation)
    if not entry or entry.get("status") != "incomplete":
        raise ValueError("Only an explicitly identified incomplete operation can be retried")
    recovery = root / "recovery" / f"{group}-{time.time_ns()}.json"
    atomic_json(recovery, {"operation": operation, "previous_receipt": document,
                          "reason": "Explicit retry-failed command; cached completed model responses remain intact"})
    del document["operations"][operation]
    document["document_digest"] = digest({key: document[key] for key in ("schema_version", "operations")})
    atomic_json(path, document)


def prepare_resume(root: Path) -> None:
    unresolved = []
    groups = json.loads((root / "control.json").read_text())["active_groups"]
    for group in groups:
        path = root / group / "search/workspace/artifacts/xlab_operations.json"
        if path.exists():
            for operation, entry in json.loads(path.read_text()).get("operations", {}).items():
                if entry.get("status") != "published":
                    unresolved.append(f"{group}:{operation}")
    if unresolved:
        raise RuntimeError("Inspect and reconcile unfinished provider operations before resume: " + ", ".join(unresolved))
    path = root / "workflow.json"
    if path.exists():
        state = json.loads(path.read_text())
        atomic_json(root / "recovery" / f"workflow-{time.time_ns()}.json", state)
        for stage, entry in list(state.get("stages", {}).items()):
            if entry.get("status") in {"failed", "running"}:
                del state["stages"][stage]
        state["status"] = "resume_ready"
        atomic_json(path, state)


def launch(runs: Path, name: str, *, resume_archived: bool = False, adopt_running: bool = False):
    registry = json.loads((runs / "sd_deployments.json").read_text())
    record = registry["deployments"][name]
    root = Path(record["root"])
    if record["status"].startswith("archived") and not resume_archived:
        raise ValueError("Archived deployments require explicit --resume-archived")
    with (runs / "sd_execution.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for other_name, other in registry["deployments"].items():
            active = live_processes(Path(other["root"]))
            if active and not (adopt_running and other_name == name):
                raise RuntimeError(f"Deployment {other_name} still has live experiment processes: {active}")
        if adopt_running:
            # A failed supervisor can leave valid arm processes running. Hold
            # the global lease and verify no other supervisor owns this run.
            with (root / "workflow.lock").open("a") as workflow_lock:
                fcntl.flock(workflow_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if resume_archived:
            check_archive(record)
            prepare_resume(root)
        register(runs, name, {"root": str(root), "status": "running", "started_at": time.time()})
        process = subprocess.Popen(["bash", str(root / "launch.engine.sh")])
        rc = process.wait()
        # A failed supervisor must not release the pool while another arm is still alive.
        while live_processes(root):
            time.sleep(5)
        register(runs, name, {"root": str(root), "status": "completed" if rc == 0 else "stopped_incomplete", "exit_code": rc})
        if rc:
            raise SystemExit(rc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=RUNS)
    commands = parser.add_subparsers(dest="command", required=True)
    save = commands.add_parser("archive")
    save.add_argument("--root", type=Path, required=True)
    save.add_argument("--name", required=True)
    check = commands.add_parser("check")
    check.add_argument("--name", required=True)
    start = commands.add_parser("launch")
    start.add_argument("--name", required=True)
    start.add_argument("--resume-archived", action="store_true")
    start.add_argument("--adopt-running", action="store_true", help="Replace a stopped supervisor while preserving this deployment's live arms")
    retry = commands.add_parser("retry-failed", help="After inspecting diagnostics, make one failed operation retryable; does not send API calls")
    retry.add_argument("--name", required=True)
    retry.add_argument("--group", choices=["A", "B"], required=True)
    retry.add_argument("--operation", required=True)
    args = parser.parse_args()
    if args.command == "archive":
        print(json.dumps(archive(args.root.resolve(), args.name, args.runs.resolve())))
    elif args.command == "check":
        record = json.loads((args.runs / "sd_deployments.json").read_text())["deployments"][args.name]
        print(json.dumps(check_archive(record)))
    elif args.command == "retry-failed":
        record = json.loads((args.runs / "sd_deployments.json").read_text())["deployments"][args.name]
        retry_failed(Path(record["root"]), args.group, args.operation)
    else:
        launch(args.runs.resolve(), args.name, resume_archived=args.resume_archived, adopt_running=args.adopt_running)


if __name__ == "__main__":
    main()
