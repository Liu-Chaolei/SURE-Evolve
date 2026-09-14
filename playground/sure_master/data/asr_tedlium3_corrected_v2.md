# Corrected TEDLIUM3 Zipformer architecture evolution

SURE-Evolve owns execution, Slurm scheduling, debugging, scoring and promotion.
Use the existing XLab idea provider for research only. Do not launch XLab
run_experiment or any additional component ablation jobs.

Train the native regular Zipformer baseline once, then run exactly six rounds
with four distinct architecture candidates per round. Each candidate starts from
scratch and completes 30 epochs on full TEDLIUM3 train with four exclusive NPUs,
FP32, seed 42, per-card max-duration 900, SpecAugment and MUSAN enabled.
MUSAN is an authorized training augmentation corpus (p=0.5, SNR 10--20).
The official Unigram-500 tokenizer and lowercase supervision preserving <unk>
are frozen resources. No tokenizer changes or extra training data are allowed.

Change structure only. Keep optimizer, loss, learning rate, augmentation, sampling,
training duration and decoding fixed. Inherit the round parent's design and code,
not its weights. Use the guarded SURE_TASK_WRAPPER --action arch interface.

All scores use greedy decoding, context size 2, at most one symbol per frame,
the corrected Ascend PackedSequence implementation and the same averaged model:
epoch=30, avg=10, use-averaged-model=true, corresponding to (epoch-20, epoch-30].
The CPU-exported average is the actual inference artifact. Never substitute the
raw last checkpoint or the cumulative average over all training history.

Regular dev (200 utterances) is the only search feedback. Do not inspect selection
or test transcripts. After six rounds, selection compares the new baseline and
the top two successful regular candidates. Test evaluates only the frozen winner
and baseline, with no retraining. Preserve full artifact provenance and report
failures honestly. No smoke, shortened training, benchmark or API test requests.
