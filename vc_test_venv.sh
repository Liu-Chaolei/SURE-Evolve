#!/usr/bin/env bash
echo "=== Checking venv paths ==="
ls -la /opt/ | grep -i whisper
echo "---"
ls -la /opt/openai__whisper-large-v3-turbo_venv/bin/python 2>&1 || echo "NO venv at openai__whisper-large-v3-turbo_venv"
echo "---"
ls -la /opt/conda/bin/python 2>&1 || echo "NO conda python"
echo "---"
which python 2>&1
echo "---"
python --version 2>&1
echo "---"
python -c "import whisper; print('whisper OK')" 2>&1
echo "---"
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())" 2>&1
echo "=== DONE ==="
