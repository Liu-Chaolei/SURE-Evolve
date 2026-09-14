"""Offline regression coverage for the corrected full ASR evolution protocol."""
import gzip
import json
import os
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace
import importlib.util
from unittest.mock import patch

import yaml

from playground.sure_master.tools.prepare_asr_corrected import normalize_transcript, rewrite_manifest
from playground.sure_master.tools.prepare_tedlium import Segment
from playground.sure_master.runtime.asr_averaging import averaged_paths, average_checkpoints_with_averaged_model
from playground.sure_master.runtime.asr_protocol import validate_resources
from playground.sure_master.core.utils.model_artifact import retain_model_artifact, restore_model_artifact


class CorrectedAsrTests(unittest.TestCase):
    def test_actual_official_tokenizer_and_training_helper_use_unknown_id(self):
        import sentencepiece as spm
        root = Path(__file__).resolve().parents[4]
        tokenizer = root / 'runs/asr_formal_6x3_npu4_20260912_131347/official_inference/assets/bpe.model'
        helper = Path('/shared/chaolei.liu/ASR/icefall/egs/tedlium3/ASR/local/convert_transcript_words_to_bpe_ids.py')
        if not tokenizer.exists() or not helper.exists():
            self.skipTest('Local official tokenizer/recipe assets are unavailable')
        spec = importlib.util.spec_from_file_location('asr_test_tokenization', helper)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sp = spm.SentencePieceProcessor(model_file=str(tokenizer))
        tokens = module.convert_texts_into_ids([normalize_transcript('<UNK> HELLO WORLD')], sp)[0]
        self.assertEqual(tokens.count(sp.unk_id()), 1)
        self.assertNotIn('<UNK>', sp.decode(tokens))

    def test_profile_is_six_by_four_with_fixed_training_and_averaging(self):
        root = Path(__file__).resolve().parents[4]
        profile = yaml.safe_load((root / 'configs/sure_master/asr-formal-6x4-npu4-v2.yaml').read_text())
        sure = profile['sure']
        self.assertEqual(sure['search_budget']['min_rounds'], 6)
        self.assertEqual(sure['search_budget']['max_rounds'], 6)
        self.assertEqual(sure['search_budget']['ideas_per_round'], 4)
        self.assertEqual(sure['slurm']['max_parallel'], 4)
        self.assertEqual(profile['session']['local']['parallel']['max_parallel'], 4)
        self.assertEqual(sure['slurm']['resource_profiles']['training']['npu'], 4)
        self.assertEqual(sure['execution_env']['SURE_ENABLE_MUSAN'], '1')
        self.assertEqual(sure['execution_env']['SURE_BASELINE_AVG'], '10')
        self.assertEqual(sure['execution_env']['SURE_BASELINE_USE_AVERAGED_MODEL'], '1')
        self.assertEqual(sure['execution_env']['SURE_USE_FP16'], '0')
        self.assertFalse(sure['execution_contract']['automatic_ablation'])
        self.assertEqual(sure['startup_mode'], 'direct_formal')
        self.assertEqual(profile['llm']['zai_flash']['model'], 'glm-5.3-flash')
        self.assertNotIn('run_experiment', ' '.join(profile['xlab']['idea_provider']['command']))

    def test_supervision_rebuild_preserves_ids_features_and_unknown_semantics(self):
        self.assertEqual(normalize_transcript(' <UNK> HELLO {NOISE} '), '<unk> hello [noise]')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / 'old.gz', root / 'new.gz'
            features = {'num_features': 80, 'sampling_rate': 16000, 'storage_path': '/shared/feats.lca'}
            row = {'id': 'talk-1', 'start': 1.0, 'duration': 2.0, 'channel': 0, 'features': features,
                   'supervisions': [{'id': 'talk-1', 'start': 0, 'text': '<UNK> HELLO'}]}
            with gzip.open(source, 'wt') as stream:
                stream.write(json.dumps(row)+'\n')
            original = source.read_bytes()
            segments = [Segment('talk-1', 'talk', 'audio.sph', 0, 1, 2, '<unk> hello', 'talk')]
            self.assertEqual(rewrite_manifest(source, target, segments), 1)
            with gzip.open(target, 'rt') as stream:
                new = json.loads(stream.readline())
            self.assertEqual(new['supervisions'][0]['text'], '<unk> hello')
            self.assertEqual(new['features'], features)
            self.assertEqual(source.read_bytes(), original)
            segments[0] = replace(segments[0], start=2)
            with self.assertRaisesRegex(ValueError, 'interval mismatch'):
                rewrite_manifest(source, target, segments)

    def test_frozen_replay_does_not_disable_structure_only_search_guard(self):
        from playground.sure_master.tools.run_icefall_zipformer_candidate import validate_candidate_args
        args = dict(candidate_type='inference', action='decode_only', train_extra_args=[], decode_extra_args=[])
        with patch.dict(os.environ, {'SURE_SEARCH_SCOPE': 'architecture_only'}, clear=True):
            with self.assertRaisesRegex(ValueError, 'Architecture-only'):
                validate_candidate_args(**args)
            validate_candidate_args(**args, frozen_replay=True)

    def test_musan_is_staged_with_train_but_not_for_inference(self):
        from playground.sure_master.runtime import staging
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            (source / 'fbank').mkdir(parents=True)
            (source / 'lang_bpe_500').mkdir()
            (source / 'lang_bpe_500/bpe.model').write_bytes(b'fixed vocabulary')
            feature = root / 'noise.lca'
            feature.write_bytes(b'features')
            for name in ['tedlium_cuts_train.jsonl.gz', 'musan_cuts.jsonl.gz']:
                with gzip.open(source / 'fbank' / name, 'wt') as stream:
                    stream.write(json.dumps({'id': name, 'features': {'storage_path': str(feature)}})+'\n')
            local = root / 'local'
            local.mkdir()
            real_path = Path
            def paths(value):
                return local if value == '/local/job' else real_path(value)
            with patch.dict(os.environ, {'SLURM_JOB_ID': 'offline', 'SURE_ENABLE_MUSAN': '1'}), patch.object(staging, 'Path', paths):
                target = staging.stage_view(source)
                with gzip.open(target / 'fbank/musan_cuts.jsonl.gz', 'rt') as stream:
                    row = json.loads(stream.readline())
                self.assertTrue(Path(row['features']['storage_path']).is_relative_to(local))
                self.assertEqual(Path(row['features']['storage_path']).read_bytes(), b'features')
                (source / 'fbank/tedlium_cuts_train.jsonl.gz').rename(source / 'fbank/tedlium_cuts_dev.jsonl.gz')
                eval_target = staging.stage_view(source)
                self.assertFalse((eval_target / 'fbank/musan_cuts.jsonl.gz').exists())

    def test_missing_or_wrong_protocol_cannot_start_formal_training(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / 'preparation.json'
            marker.write_text(json.dumps({'protocol': 'old-bpe', 'features_ready': True}))
            env = {'SURE_ASR_PROTOCOL': 'tedlium3.unigram500.musan.v2', 'SURE_ASR_PREPARATION': str(marker),
                   'SURE_MAX_TRAIN_EPOCHS': '30', 'SURE_USE_FP16': '0', 'SURE_ENABLE_MUSAN': '1',
                   'SURE_BASELINE_AVG': '10', 'SURE_BASELINE_USE_AVERAGED_MODEL': '1',
                   'SURE_ASR_CPU_AVERAGING': '1', 'SURE_ASR_FIXED_SEED': '42'}
            with self.assertRaisesRegex(ValueError, 'prepared-data protocol'):
                validate_resources(env)
            env['SURE_ENABLE_MUSAN'] = '0'
            with self.assertRaisesRegex(ValueError, 'overridden'):
                validate_resources(env)

    def test_cpu_average_is_last_window_and_survives_cleanup_and_replay(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / 'models/zipformer_candidate'
            models.mkdir(parents=True)
            (root / 'data/lang_bpe_500').mkdir(parents=True)
            bpe = root / 'data/lang_bpe_500/bpe.model'
            bpe.write_bytes(b'fixed test vocabulary')
            for epoch, weight in [(1, 0), (20, 2), (30, 4)]:
                torch.save({'model_avg': {'weight': torch.tensor([weight, weight+2], dtype=torch.float64)},
                            'batch_idx_train': epoch*100, 'average_period': 200}, models / f'epoch-{epoch}.pt')
            env = {'SURE_ASR_AVERAGING_TOKENIZER': str(bpe), 'SURE_ASR_DATA_FINGERPRINT': 'fixed-data'}
            with patch.dict(os.environ, env):
                state = average_checkpoints_with_averaged_model(models/'epoch-20.pt', models/'epoch-30.pt')
                torch.testing.assert_close(state['weight'], torch.tensor([8., 10.]))
                with patch('icefall.checkpoint.average_checkpoints_with_averaged_model', side_effect=AssertionError('must reuse')):
                    torch.testing.assert_close(average_checkpoints_with_averaged_model(models/'epoch-20.pt', models/'epoch-30.pt')['weight'], state['weight'])
                averaged, receipt = averaged_paths(models, 30, 10)
                (root/'artifacts').mkdir()
                changes = {'inference_config': {'actual_bpe_model': str(bpe), 'decode_avg': 10,
                            'use_averaged_model': True, 'averaging': json.loads(receipt.read_text())},
                           'produced_artifacts': {'inference_checkpoint': str(averaged)}}
                (root/'artifacts/candidate_changes.json').write_text(json.dumps(changes))
                result = retain_model_artifact(root)
                kept = Path(result['checkpoint_dir'])
                self.assertTrue((kept/'epoch-20.pt').is_file())
                self.assertTrue((kept/'epoch-30.pt').is_file())
                self.assertFalse((kept/'epoch-1.pt').exists())
                self.assertTrue(Path(result['inference_checkpoint']).is_file())
                replay = root/'replay'
                replay.mkdir()
                restored = replay/'models/zipformer_candidate'
                epoch, restored_bpe, metadata = restore_model_artifact(result['model_artifact'], restored, workspace=replay)
                self.assertEqual(epoch, 30)
                self.assertEqual(metadata['inference_config']['decode_avg'], 10)
                with patch.dict(os.environ, {'SURE_ASR_AVERAGING_TOKENIZER': str(restored_bpe)}):
                    torch.testing.assert_close(average_checkpoints_with_averaged_model(restored/'epoch-20.pt', restored/'epoch-30.pt')['weight'], state['weight'])
                bpe.write_bytes(b'wrong tokenizer')
                with self.assertRaisesRegex(ValueError, 'binding or digest'):
                    average_checkpoints_with_averaged_model(models/'epoch-20.pt', models/'epoch-30.pt')

    def test_complete_training_is_required_to_skip_training_on_decode_retry(self):
        import torch
        from playground.sure_master.runtime.resume import complete_epoch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = {'cur_epoch': 30, 'batch_idx_train': 100, 'model': {}, 'optimizer': {}, 'scheduler': {}}
            torch.save(state, root/'epoch-30.pt')
            self.assertTrue(complete_epoch(root, 30))
            self.assertFalse(complete_epoch(root, 29))
            del state['optimizer']
            torch.save(state, root/'epoch-30.pt')
            self.assertFalse(complete_epoch(root, 30))


if __name__ == '__main__':
    unittest.main()
