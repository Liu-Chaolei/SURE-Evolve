"""CPU averaging and a reusable, fingerprinted ASR inference checkpoint."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def averaged_paths(directory: Path, epoch: int, avg: int) -> tuple[Path, Path]:
    model = directory / f'averaged-model-epoch-{epoch}-avg-{avg}.pt'
    return model, model.with_suffix('.json')


def average_checkpoints_with_averaged_model(filename_start, filename_end, device=None):
    """Match icefall averaging semantics, then reuse exactly the scored weights."""
    import torch
    from icefall.checkpoint import average_checkpoints_with_averaged_model as average

    start, end = Path(filename_start), Path(filename_end)
    first, last = int(start.stem.split('-')[-1]), int(end.stem.split('-')[-1])
    if first < 1 or last <= first or start.parent.resolve() != end.parent.resolve():
        raise ValueError('Invalid averaged checkpoint window')
    tokenizer = Path(os.environ['SURE_ASR_AVERAGING_TOKENIZER'])
    binding = {'schema_version': 'sure.asr_average.v1', 'start_epoch_exclusive': first,
        'end_epoch_inclusive': last, 'avg': last-first, 'use_averaged_model': True,
        'source_checkpoints': {start.name: file_digest(start), end.name: file_digest(end)},
        'tokenizer_sha256': file_digest(tokenizer),
        'data_fingerprint': os.environ.get('SURE_ASR_DATA_FINGERPRINT')}
    output, receipt = averaged_paths(end.parent, last, last-first)
    if output.exists() or receipt.exists():
        if not output.is_file() or not receipt.is_file():
            # A previous attempt may have been interrupted between the two atomic writes.
            output.unlink(missing_ok=True)
            receipt.unlink(missing_ok=True)
        else:
            previous = json.loads(receipt.read_text())
            if previous['binding'] != binding or file_digest(output) != previous['sha256']:
                raise ValueError('Averaged inference checkpoint binding or digest changed')
            print(f'[asr-average] reusing {output}', flush=True)
            return torch.load(output, map_location='cpu', weights_only=False)['model']
    state = average(str(start), str(end), device=torch.device('cpu'))
    state = {name: value.float() if value.is_floating_point() else value for name, value in state.items()}
    temporary = output.with_suffix('.pending')
    torch.save({'model': state, 'averaging': binding}, temporary)
    temporary.replace(output)
    temporary_receipt = receipt.with_suffix('.pending.json')
    temporary_receipt.write_text(json.dumps({'binding': binding, 'checkpoint': output.name,
        'sha256': file_digest(output)}, indent=2) + '\n')
    temporary_receipt.replace(receipt)
    print(f'[asr-average] exported {output}', flush=True)
    return state
