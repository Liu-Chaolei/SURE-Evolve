# Pinned Evaluation Engine Contract v1

The implementation source of record is the standalone repository declared in
`config/evaluation_engine.lock.yaml`. The lock pins repository origin, commit,
selected source-tree archive SHA256, Python executable, and CLI entrypoint.

`sure_eval.evaluation_engine.EvaluationEngine` verifies the pin on every run.
It executes `agent plan`, `metric describe`, and `metric run --validate-env` and
stores a trace for each subprocess. All task aliases supported by the pinned
engine are supported by the bridge.

Main flow must prepare `sure.eval.evaluation_input_manifest.v1` files with
checksum-bound role inputs. The bridge refuses unknown roles, missing required
roles, a pipeline that cannot run, non-JSON CLI output, non-zero commands, and
pipeline identity disagreement.

`scripts/evaluate_predictions.py` and the sandbox's copied evaluation package
are legacy compatibility surfaces. They are not permitted in a v2 agent run.
