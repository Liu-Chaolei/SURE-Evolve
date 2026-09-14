# SD research experiment

Improve DiariZen on AMI using the supplied research context. Explore structure,
training methods, inference strategies and coordinated changes without quotas.
Execute one candidate per assigned idea; do not run additional component ablations.
Search lasts exactly six rounds, four candidate slots per round. Missing or failed
candidates consume slots. The only scientific feedback is search DER.

Use SURE_TASK_WRAPPER --action prepare_source --candidate-source working/sd_candidate,
edit that workspace source, then --action candidate --candidate-source
working/sd_candidate --parameters-json with requires_training and only changed
architecture/training/inference entries. Source preparation inherits the supplied
parent's design. Training always initializes the SSL backbone from official
WavLM-Base+, initializes the rest afresh and never continues a parent optimizer.
The public powerset output and RTTM interfaces remain compatible.

Model and inference source are editable. Dataset traversal, validation, the
trusted trainer loop, scoring and training duration are fixed. Use optional
diarizen/sure_candidate.py functions for training interventions:
build_optimizers(model, training) returns a dict with wavlm and network optimizers;
build_schedulers(optimizers, training, total_updates) returns named schedulers;
transform_batch(batch) returns the training batch;
training_loss(model, batch, training) returns a differentiable scalar loss.
Schedulers step once per optimizer update and their state is checkpointed.
Validation never calls candidate loss or augmentation hooks.

Maintain seed 3407, four NPU ranks, FP32, official initialization, fixed data
splits, at most 100 epochs, validation-loss patience 10 and best-five checkpoint
averaging. Training parameters that may vary: learning_rate_wavlm,
learning_rate_network, freeze_wavlm, candidate_options (hook-specific JSON).
Do not alter other training controls. Inference-only candidates reuse the supplied
frozen parent and may only edit inference source/configuration.

Use only the task material, source APIs and evidence supplied to the current
candidate. Do not read other experiment directories, controller metric/history
files, literature stores, selection/holdout references or external data. Tools for
reading model source must not be repurposed for retrieving research history.
Report implementation failures honestly. Never invent experimental results.
