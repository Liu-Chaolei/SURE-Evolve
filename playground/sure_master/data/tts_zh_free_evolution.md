# F5-TTS Chinese free evolution

Improve Chinese speech generation from F5TTS_v1_Base. The objective is fixed SURE
Paraformer Chinese CER. Use the complete Premium train split, 50 full epochs,
official pretrained initialization with a fresh optimizer, eight allocated NPUs,
the deployment's declared training precision and final EMA weights for every weight-updating solution.
The current deployment uses BF16 autocast with FP32 master parameters, optimizer state and inference,
fused AdamW, expandable_segments, and a default budget of 25600 frames per NPU.
Search and selection
are disjoint Premium groups; Seed Chinese is used only for the final frozen test.

XLab freely chooses three distinct, evidence-backed complete solutions per round.
There are no architecture/training/inference quotas. A hypothesis may combine model,
loss, optimization, augmentation and inference changes. Identify what requires weight
updates, explain the mechanism, and use previous successes and failures as evidence.
Do not replace a scientific mechanism with a mere numeric sweep. Each round uses the
same frozen parent for inference; training always starts from the official weights.

Use SURE_TASK_WRAPPER --action candidate --parameters-json with requires_training
and any inference/training/architecture objects. For code changes, first call
--action prepare_source --candidate-source working/f5_candidate, edit that copy,
then pass --candidate-source working/f5_candidate to the candidate action.
Model/loss code and inference code are editable. The framework owns the trainer
loop, data traversal, 50-epoch completion and scoring. Training extensions belong
in src/f5_tts/sure_candidate.py: build_optimizer(parameters, training),
build_scheduler(optimizer, total_updates, training), transform_batch(batch).
Keep vocabulary, 24 kHz mel interface, vocoder and evaluation targets fixed.

All artifacts must be real and replayable. Never modify shared F5 or SURE sources,
train on evaluation data, synthesize missing scores, or silently shorten training.
