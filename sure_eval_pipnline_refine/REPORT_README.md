# Evaluation Report

## Result

- Dataset: `librispeech_test-clean__v1.0.1`
- Evaluated rows: 3 of 2619
- Task/language: ASR / English
- Metric: WER
- Score: `0.0` (`0.000000%`)
- Prediction validation: passed, 3/3
- Protocol: `strict_core`
- Status: success

This is a bounded chain-validation result, not a publishable full LibriSpeech
test-clean benchmark score.

## Report Naming

The formal evaluation report is `report.jsonl`, with one row per dataset and
metric. `main_agent_run_report.json` is only the orchestration summary; it is
not the metric report and should not be presented as the evaluation result.

Run directory:

`/hpc_stor03/project/oref/nfs/models/openai__whisper-large-v3-turbo/eval_runs/main_agent_openai_whisper_librispeech_test-clean_3sample_20260808_001`

Primary files:

- `report.jsonl`: formal dataset-metric report
- `evaluation_payload.json`: structured evaluation payload
- `protocol.yaml`: executed protocol and inputs
- `metrics/librispeech_test-clean__v1.0.1/wer/report.json`: detailed metric trace
- `metrics/librispeech_test-clean__v1.0.1/wer/pipeline_description.json`: route and node description
- `sample_reports/librispeech_test-clean__v1.0.1/wer.jsonl`: sample-level results
- `report_snapshot.md`: human-readable generated snapshot
- `main_agent_run_report.json`: main-agent orchestration evidence

The report identity check passed without changing canonical dataset-ID or
report naming behavior.
