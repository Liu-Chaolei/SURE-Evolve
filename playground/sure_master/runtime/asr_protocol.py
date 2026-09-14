"""Validate the corrected ASR inputs before training or frozen replay."""
from __future__ import annotations

import json
from pathlib import Path

from .asr_averaging import file_digest


def validate_resources(env) -> dict | None:
    protocol = env.get('SURE_ASR_PROTOCOL')
    if not protocol:
        return None
    fixed = {'SURE_MAX_TRAIN_EPOCHS': '30', 'SURE_USE_FP16': '0', 'SURE_ENABLE_MUSAN': '1',
             'SURE_BASELINE_AVG': '10', 'SURE_BASELINE_USE_AVERAGED_MODEL': '1',
             'SURE_ASR_CPU_AVERAGING': '1', 'SURE_ASR_FIXED_SEED': '42'}
    if any(str(env.get(key)) != value for key, value in fixed.items()):
        raise ValueError('The corrected ASR training/averaging protocol was overridden')
    marker = Path(env['SURE_ASR_PREPARATION'])
    prepared = json.loads(marker.read_text())
    if (protocol != 'tedlium3.unigram500.musan.v2' or prepared.get('protocol') != protocol
            or not prepared.get('features_ready') or prepared.get('training_selection') != 'full'
            or prepared.get('fingerprint') != env.get('SURE_ASR_DATA_FINGERPRINT')):
        raise ValueError('Corrected ASR prepared-data protocol does not match the frozen run')
    if prepared.get('splits', {}).get('train', {}).get('utterances') != 268263:
        raise ValueError('The complete TEDLIUM3 training split is required')
    files = prepared.get('files', {})
    required = {'lang_bpe_500/bpe.model', 'fbank/musan_cuts.jsonl.gz',
                'fbank/tedlium_cuts_train.jsonl.gz', 'fbank/tedlium_cuts_dev.jsonl.gz',
                'fbank/tedlium_cuts_test.jsonl.gz'}
    if not required.issubset(files):
        raise ValueError('Corrected data is missing validated tokenizer, features or MUSAN')
    for relative, expected in files.items():
        path = marker.parent / relative
        if Path(relative).is_absolute() or '..' in Path(relative).parts or file_digest(path) != expected:
            raise ValueError(f'Prepared ASR resource changed: {relative}')
    for path, expected in prepared.get('identity', {}).get('feature_files', {}).items():
        stat = Path(path).stat()
        if stat.st_size != expected['size'] or stat.st_mtime_ns != expected['mtime_ns']:
            raise ValueError(f'Reused feature shard changed: {path}')
    return prepared
