"""Explicit, exact-dependency imports of previously completed mode outputs."""
import hashlib
import json
from pathlib import Path
from ..checkpoints import stage_dependency_signature


def compatible_mode(manifest_path, mode, dependencies, output_dir):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != 'xlab.mode-migration.v1':
        raise ValueError('Unknown mode migration schema')
    entry = manifest['modes'].get(mode)
    receipt = {'mode':mode,'status':'not_imported','reason':'no eligible source mode'}
    output = None
    if entry:
        root = Path(manifest['source_run']).resolve()
        path = (root / entry['path']).resolve()
        path.relative_to(root)
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry['sha256']:
            raise ValueError('Mode import artifact changed')
        output = json.loads(raw)
        if entry['dependency_signature'] != stage_dependency_signature('search.'+mode, dependencies):
            output = None
            receipt['reason'] = 'search dependencies differ; recompute rather than relabel provenance'
        elif any(b['error_type'] in {'ProviderError','ProviderExhaustedError','ComponentNoveltyError'} for b in output.get('rollout_blockers',[])):
            output = None
            receipt['reason'] = 'source mode contains unresolved infrastructure or novelty blockers'
        else:
            receipt.update(status='imported',reason='exact stage dependencies and source digest verified',source=str(path),sha256=entry['sha256'])
            output['metadata']['migration_origin'] = dict(receipt)
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    path=output_dir / f'migration.{mode}.json'
    path.write_text(json.dumps(receipt,indent=2)+'\n')
    return output
