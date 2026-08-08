# SURE 语音任务指标汇总

## 任务与实际主指标

| 任务 | 默认主指标 | 默认 pipeline / route | 说明 |
| --- | --- | --- | --- |
| ASR, 中文 | `CER` | `asr.zh.cer.aispeech_norm.wenet_cer` | 中文 ASR 默认用 AISpeech 归一化后计算 WeNet CER。 |
| ASR, 英文 | `WER` | `asr.en.wer.whisper_norm.wenet_wer` | 英文 ASR 默认用 Whisper English normalization 后计算 WeNet WER。 |
| ASR, 中英混合 | `MER` | `asr.cs.mer.aispeech_norm.wenet_mer` | code-switch ASR 默认计算 mixed error rate。 |
| S2TT | `BLEU` | `s2tt.{tokenizer_profile}.bleu.sacrebleu` | 默认主指标是 SacreBLEU BLEU；tokenizer profile 由语言或调用参数决定。 |
| KWS | `accuracy` | `kws.{profile}.accuracy.wekws_det` | pipeline 只接受 `accuracy` 作为 primary metric；DET 相关统计会放在 details 中。 |
| SLU | `accuracy` | `normalization/prompt_norm` -> `scoring/classify` | 对 prompt choices 做归一化后按分类准确率计分。 |
| CLASSIFICATION | `accuracy` | `scoring/classify` | 通用分类任务默认准确率；SER 和 GR 走该路由并使用兼容 label spec。 |
| SER | `accuracy` | `classification.accuracy.classify` | 语音情感识别在 SURE 中按分类准确率评测。 |
| GR | `accuracy` | `classification.accuracy.classify` | 性别识别在 SURE 中按分类准确率评测。 |
| SD | `DER` | `scoring/meeteval` | speaker diarization 默认 DER，默认 collar 为 `0.25`。 |
| SA-ASR | `cpWER` | `scoring/meeteval` | speaker-attributed ASR 默认主指标 cpWER；同时报告 companion `DER`，默认 collar 为 `0.5`。 |
| TTS, 中文 | `tts_cer` | `tts.zh.tts_cer.funasr_loader_16k_mono.paraformer_zh.asr_cer` | 先用 Paraformer 转写生成音频，再按 ASR CER 评估可懂度。 |
| TTS, 英文 | `tts_wer` | `tts.en.tts_wer.whisper_large_v3.whisper_norm.wenet_wer` | 先用 Whisper large-v3 转写生成音频，再做 Whisper normalization 和 WER。 |
| VC, 中文 | `vc_cer` | `vc.zh.vc_cer.funasr_loader_16k_mono.paraformer_zh.asr_cer` | 默认评估转换音频内容保持度；中文走 Paraformer + CER。 |
| VC, 英文 | `vc_wer` | `vc.en.vc_wer.whisper_large_v3.whisper_norm.wenet_wer` | 默认评估转换音频内容保持度；英文走 Whisper large-v3 + WER。 |

## 指标实现位置

路径前缀：`/hpc_stor03/sjtu_home/chaolei.liu/sure/src/sure_eval/evaluation/`

| 任务 / 指标 | 任务入口 | 实际计算实现 |
| --- | --- | --- |
| ASR `CER` | `tasks/asr/pipeline.py:evaluate_asr_files` | `nodes/scoring/wenet_wer/node.py:score_wenet_cer`; edit distance 核心在 `nodes/scoring/wenet_wer/wenet_compute_cer.py:compute_wer`。 |
| ASR `WER` | `tasks/asr/pipeline.py:evaluate_asr_files` | `nodes/scoring/wenet_wer/node.py:score_wenet_wer`; edit distance 核心在 `nodes/scoring/wenet_wer/wenet_compute_cer.py:compute_wer`。 |
| ASR `MER` | `tasks/asr/pipeline.py:evaluate_asr_files` | `nodes/scoring/wenet_wer/node.py:score_codeswitch_mer`; 内部复用 `compute_wer` 分别计算 MER/CER/WER。 |
| ASR 默认归一化 | `tasks/asr/pipeline.py:_normalization_node` | 中文/混合：`nodes/normalization/aispeech_norm/node.py:normalize_asr_files` 和 `normalize_codeswitch_asr_files`；英文：`nodes/normalization/whisper_norm/node.py:normalize_whisper_asr_files`。 |
| S2TT `BLEU` | `tasks/s2tt/pipeline.py:evaluate_s2tt_files` | `nodes/scoring/sacrebleu/node.py:score_sacrebleu`。 |
| KWS `accuracy` | `tasks/kws/pipeline.py:evaluate_kws_files` 和 `evaluate_kws_samples` | `nodes/scoring/wekws_det/node.py:score_wekws_det`; 阈值准确率在 `nodes/scoring/wekws_det/metrics.py:KWSMetric.calculate_samples`。 |
| SLU `accuracy` | `tasks/slu/pipeline.py:evaluate_slu_files` | prompt 归一化在 `nodes/normalization/prompt_norm/node.py:normalize_prompt_choice_files`; 分类计分在 `nodes/scoring/classify/node.py:score_classification_rows`。 |
| CLASSIFICATION / SER / GR `accuracy` | `tasks/classification/pipeline.py:evaluate_classification_files` | `nodes/scoring/classify/node.py:score_classification_files` 和 `score_classification_rows`; SER/GR 默认 label spec 在 `default_label_spec`。 |
| SD `DER` | `tasks/sd/pipeline.py:evaluate_sd_files` | `nodes/scoring/meeteval/node.py:score_meeteval`; DER 分支在 `_score_der`，调用 `meeteval.der.dscore`。 |
| SA-ASR `cpWER` | `tasks/sa_asr/pipeline.py:evaluate_sa_asr_files` | `nodes/scoring/meeteval/node.py:score_meeteval`; cpWER 分支在 `_score_cpwer`，调用 `meeteval.wer.cpwer` 和 `combine_error_rates`。 |
| TTS `tts_cer` | `tasks/tts/pipeline.py:evaluate_tts_samples` 和 `_evaluate_semantic` | 中文转写：`nodes/frontend/funasr_loader_16k_mono/node.py:describe_funasr_loader_16k_mono` + `nodes/transcription/paraformer_zh/node.py:transcribe_paraformer_zh`; 计分复用 `nodes/transcription/common/audio_semantic.py:score_transcripts_with_asr` -> `tasks/asr/pipeline.py:evaluate_asr_files` -> `score_wenet_cer`。 |
| TTS `tts_wer` | `tasks/tts/pipeline.py:evaluate_tts_samples` 和 `_evaluate_semantic` | 英文转写：`nodes/transcription/whisper_large_v3/node.py:transcribe_whisper_large_v3`; 计分复用 `score_transcripts_with_asr` -> ASR `score_wenet_wer`。 |
| VC `vc_cer` | `tasks/vc/pipeline.py:evaluate_vc_samples` 和 `_evaluate_semantic` | 中文转写：`describe_funasr_loader_16k_mono` + `transcribe_paraformer_zh`; 计分复用 `score_transcripts_with_asr` -> ASR `score_wenet_cer`。 |
| VC `vc_wer` | `tasks/vc/pipeline.py:evaluate_vc_samples` 和 `_evaluate_semantic` | 英文转写：`transcribe_whisper_large_v3`; 计分复用 `score_transcripts_with_asr` -> ASR `score_wenet_wer`。 |

## 关键依据文件

- `tasks/asr/manifest.yaml`
- `tasks/s2tt/manifest.yaml`
- `tasks/kws/pipeline.py`
- `tasks/slu/manifest.yaml`
- `tasks/classification/manifest.yaml`
- `tasks/sd/manifest.yaml`
- `tasks/sa_asr/manifest.yaml`
- `tasks/tts/manifest.yaml`
- `tasks/tts/pipeline.py`
- `tasks/vc/manifest.yaml`
- `tasks/vc/pipeline.py`
- `cli_adapters.py`
