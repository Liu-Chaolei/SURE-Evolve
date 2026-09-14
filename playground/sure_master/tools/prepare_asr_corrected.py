"""Prepare the full TEDLIUM/Unigram/MUSAN protocol without rewriting old data."""
from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile

from playground.sure_master.tools.prepare_tedlium import corpus_directory, prepare_refs, read_stm_split

PROTOCOL = "tedlium3.unigram500.musan.v2"
TOKENIZER_SHA256 = "f950ca4200a0611ae3a2b2cb561f34ed7f39ae554512dce54134c55aa29d7188"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def normalize_transcript(text: str) -> str:
    # Do not apply tokenizer-training vocabulary filters to ASR supervision.
    return text.strip().lower().replace("{noise}", "[noise]")


def rewrite_manifest(source: Path, destination: Path, segments: list) -> int:
    expected = {segment.key: segment for segment in segments}
    seen = set()
    pending = destination.with_suffix(".pending.gz")
    with gzip.open(source, "rt") as reader, gzip.open(pending, "wt") as writer:
        for line in reader:
            cut = json.loads(line)
            key = cut["id"]
            if key not in expected or key in seen:
                raise ValueError(f"Unexpected or duplicate feature cut: {key}")
            segment = expected[key]
            if abs(cut['start'] - segment.start) > 1e-6 or abs(cut['duration'] - segment.duration) > 1e-6:
                raise ValueError(f"Feature/STM interval mismatch: {key}")
            if cut['channel'] != segment.channel or len(cut['supervisions']) != 1:
                raise ValueError(f"Feature/STM channel or supervision mismatch: {key}")
            features = cut.get('features', {})
            if features.get('num_features') != 80 or features.get('sampling_rate') != 16000:
                raise ValueError(f"Incompatible Fbank: {key}")
            supervision = cut['supervisions'][0]
            if supervision['id'] != key or supervision['start'] != 0:
                raise ValueError(f"Unexpected supervision binding: {key}")
            supervision['text'] = normalize_transcript(segment.text)
            writer.write(json.dumps(cut, ensure_ascii=False) + '\n')
            seen.add(key)
    if seen != set(expected):
        raise ValueError(f"Missing {len(set(expected) - seen)} source utterances")
    pending.replace(destination)
    return len(seen)


def long_noise(cut) -> bool:
    return cut.duration > 5


def prepare_musan(archive: Path, resources: Path, jobs: int) -> Path:
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from lhotse import CutSet, Fbank, FbankConfig, LilcomChunkyWriter, combine
    from lhotse.recipes.musan import prepare_musan as manifests_for_musan

    resources.mkdir(parents=True, exist_ok=True)
    identity = {'archive_sha256': digest(archive), 'fbank': 'kaldi-80', 'window_seconds': 10,
                'minimum_duration_exclusive': 5, 'lhotse_version': __import__('lhotse').__version__}
    receipt = resources / 'preparation.json'
    manifest = resources / 'fbank/musan_cuts.jsonl.gz'
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous['identity'] != identity:
            raise ValueError('MUSAN resource identity changed; use a new resource directory')
        if previous.get('complete') and manifest.is_file() and digest(manifest) == previous['manifest_sha256']:
            return manifest
    raw = resources / 'musan'
    if not (resources / 'extracted.json').exists():
        print('Extracting MUSAN', flush=True)
        with tarfile.open(archive) as stream:
            stream.extractall(resources, filter='data')
        write_json(resources / 'extracted.json', identity)
    manifests = manifests_for_musan(raw, output_dir=resources / 'manifests')
    (resources / 'fbank').mkdir(exist_ok=True)
    print('Computing MUSAN Fbank', flush=True)
    cuts = (CutSet.from_manifests(recordings=combine(part['recordings'] for part in manifests.values()))
            .cut_into_windows(10.0).filter(long_noise)
            .compute_and_store_features(Fbank(FbankConfig(num_mel_bins=80)),
                storage_path=str(resources / 'fbank/musan_feats'), num_jobs=jobs, storage_type=LilcomChunkyWriter))
    cuts.to_file(manifest)
    write_json(receipt, {'identity': identity, 'complete': True, 'cuts': len(cuts),
                        'manifest_sha256': digest(manifest)})
    return manifest


def prepare(args) -> dict:
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Full resource preparation requires a Slurm job')
    source, output = args.feature_data.resolve(), args.output.resolve()
    if output == source or source in output.parents:
        raise ValueError('Use an independent version directory')
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.preparation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if digest(args.tokenizer) != TOKENIZER_SHA256:
            raise ValueError('Expected the verified official Unigram-500 tokenizer')
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
        if sp.get_piece_size() != 500 or sp.unk_id() != 2 or sp.piece_to_id('<blk>') != 0:
            raise ValueError('Unexpected tokenizer symbol contract')
        corpus = corpus_directory(args.corpus)
        splits = {name: read_stm_split(corpus, name) for name in ('train', 'dev', 'test')}
        expected_counts = {'train': 268263, 'dev': 507, 'test': 1155}
        if {name: len(rows) for name, rows in splits.items()} != expected_counts:
            raise ValueError('This protocol requires the complete original TEDLIUM3 splits')
        noise = prepare_musan(args.musan_archive, args.musan_resources.resolve(), args.jobs)
        feature_files = {}
        for name in splits:
            print(f'Fingerprinting reused {name} feature shards', flush=True)
            shards = sorted((source / 'fbank' / f'tedlium_feats_{name}').glob('*.lca'))
            if not shards:
                raise ValueError(f'No reusable feature shards for {name}')
            for path in shards:
                stat = path.stat()
                feature_files[str(path.resolve())] = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                                                     'sha256': digest(path)}
        for path in sorted((noise.parent / 'musan_feats').glob('*.lca')):
            stat = path.stat()
            feature_files[str(path.resolve())] = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                                                 'sha256': digest(path)}
        identity = {'protocol': PROTOCOL, 'normalization': 'lowercase_preserve_unk.v1',
            'feature_files': feature_files,
            'tokenizer_sha256': TOKENIZER_SHA256, 'musan_manifest_sha256': digest(noise),
            'source_manifests': {name: digest(source / 'fbank' / f'tedlium_cuts_{name}.jsonl.gz') for name in splits},
            'supervision_sha256': hashlib.sha256(json.dumps({name: [(s.key, s.start, s.duration, normalize_transcript(s.text))
                for s in rows] for name, rows in splits.items()}, sort_keys=True).encode()).hexdigest()}
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        marker = output / 'preparation.json'
        if marker.exists():
            existing = json.loads(marker.read_text())
            if existing['fingerprint'] != fingerprint:
                raise ValueError('Prepared protocol changed; choose a new output version')
        write_json(marker, {'protocol': PROTOCOL, 'fingerprint': fingerprint, 'features_ready': False})
        (output / 'fbank').mkdir(exist_ok=True)
        (output / 'lang_bpe_500').mkdir(exist_ok=True)
        shutil.copy2(args.tokenizer, output / 'lang_bpe_500/bpe.model')
        files = {}
        for name, rows in splits.items():
            path = output / 'fbank' / f'tedlium_cuts_{name}.jsonl.gz'
            count = rewrite_manifest(source / 'fbank' / path.name, path, rows)
            files[str(path.relative_to(output))] = digest(path)
            print(f'Rebuilt {name} supervision: {count}', flush=True)
            target = output / 'fbank' / f'tedlium_feats_{name}'
            if not target.exists():
                target.symlink_to(source / 'fbank' / target.name, target_is_directory=True)
        noise_target = output / 'fbank/musan_cuts.jsonl.gz'
        if not noise_target.exists():
            noise_target.symlink_to(noise)
        files['fbank/musan_cuts.jsonl.gz'] = digest(noise_target)
        files['lang_bpe_500/bpe.model'] = TOKENIZER_SHA256
        ref_counts = prepare_refs(splits, output / 'refs')
        for path in (output / 'refs').glob('*.txt'):
            if path.name in {'asr_tedlium3_regular_ref.txt', 'asr_tedlium3_selection_ref.txt', 'asr_tedlium3_test_ref.txt'}:
                if path.read_bytes() != (source / 'refs' / path.name).read_bytes():
                    raise ValueError(f'Frozen reference changed: {path.name}')
                files[str(path.relative_to(output))] = digest(path)
        summary = {'protocol': PROTOCOL, 'fingerprint': fingerprint, 'identity': identity,
            'corpus': str(corpus), 'features_ready': True, 'training_selection': 'full', 'requested_train_hours': 0,
            'feature_reuse': True, 'feature_source': str(source), 'refs': ref_counts, 'files': files,
            'augmentation': {'musan': True, 'probability': 0.5, 'snr': [10, 20], 'speed_perturbation': False},
            'splits': {name: {'utterances': len(rows), 'hours': sum(s.duration for s in rows)/3600} for name, rows in splits.items()}}
        write_json(marker, summary)
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--feature-data', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--musan-archive', type=Path, required=True)
    parser.add_argument('--musan-resources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=16)
    print(json.dumps(prepare(parser.parse_args()), indent=2))


if __name__ == '__main__':
    main()
