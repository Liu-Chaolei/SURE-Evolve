"""Copy the complete declared local resource bundle, never evidence excerpts."""
from __future__ import annotations

import shutil
from copy import deepcopy
from pathlib import Path

from research_idea_lib.common import atomic_write_json
from research_idea_lib.inputs import IdeaRequest
from research_idea_lib.resources.manifest import sha256_file
from research_idea_lib.survey_repository import SurveyArtifactRepository, resolve_artifact_path


def freeze_survey(survey: Path, destination: Path, cwd: Path) -> dict:
    repository = SurveyArtifactRepository.from_request(IdeaRequest(survey_path=survey), cwd)
    if not repository.validation().get("passed") or not repository.source.resources.passed:
        raise ValueError("Survey and complete resource bundle must pass validation before freezing")
    source = repository.source
    source_root = source.manifest_path.parent.resolve()
    if destination.exists():
        raise ValueError("Survey snapshot destination already exists")
    files = {source.manifest_path, source.survey_json_path, source.survey_md_path,
             source.citations_path, source.report_path, source.resources.manifest_path}
    manifest = deepcopy(source.manifest)
    for artifact in manifest.get("artifacts", []):
        if artifact.get("path"):
            path = resolve_artifact_path(artifact["path"], cwd, source_root)
            artifact["path"] = path.relative_to(source_root).as_posix()
            files.add(path)
    bundle = source.resources.bundle
    for descriptor in bundle.manifest.resources:
        resource_root = bundle.path_for(descriptor.resource_id)
        files.update(resource_root / item.path for item in descriptor.files)
    hashes = {}
    for source_path in sorted(path for path in files if path is not None):
        relative = source_path.resolve().relative_to(source_root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target)
        hashes[relative.as_posix()] = sha256_file(target)
    atomic_write_json(destination / "manifest.json", manifest)
    hashes["manifest.json"] = sha256_file(destination / "manifest.json")
    exported_survey = destination / source.survey_json_path.resolve().relative_to(source_root)
    exported = SurveyArtifactRepository.from_request(IdeaRequest(survey_path=exported_survey), destination)
    if not exported.validation().get("passed") or not exported.source.resources.passed:
        raise ValueError("Relocated survey failed validation; source must use portable artifact paths")
    atomic_write_json(destination / "bundle_snapshot.json", {"schema_version": "sure.survey_snapshot.v1", "files": hashes})
    return {"survey_path": str(exported_survey), "files": len(hashes)}
