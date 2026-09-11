# Chinese F5-TTS Architecture Search with SURE

Optimize the Chinese F5-TTS model on the SURE `tts_zh_cer` task.

- Language: Chinese (`zh`)
- Primary metric: `tts_cer`, lower is better
- Training data: the prepared Premium train split only
- Search data: the prepared Premium search split
- Final holdout: the prepared Seed-TTS zh split, used only after search
- Base model: the official F5TTS v1 Base checkpoint

Every XLab research idea must be an architecture-only change. The candidate
must use `SURE_TASK_WRAPPER --action arch --parameters-json`, keep training and
inference settings fixed, and complete the configured full training budget.
The implementation must not change the tokenizer, vocabulary, mel settings,
vocoder, dataset assignment, or evaluation protocol.
