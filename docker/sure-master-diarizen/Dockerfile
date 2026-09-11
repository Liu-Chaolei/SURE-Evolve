FROM registry.cluster.local:5000/users/chaolei-liu/sure-zipformer@sha256:d557a61e7beb3b675917946061e63790997fc9f672563fe7882f3336b1a8fa64
USER root
COPY constraints.txt /tmp/sd-constraints.txt
COPY requirements.txt /tmp/sd-requirements.txt
RUN python -m pip install --no-cache-dir -c /tmp/sd-constraints.txt -r /tmp/sd-requirements.txt
COPY source /opt/diarizen
RUN python -m pip install --no-cache-dir --no-deps /opt/diarizen /opt/diarizen/pyannote-audio
RUN python -m venv /opt/sure-metrics && /opt/sure-metrics/bin/pip install --no-cache-dir pydantic pydantic-settings PyYAML regex rich requests httpx numpy scipy pandas meeteval soundfile librosa huggingface-hub structlog
COPY md-eval-22.pl /usr/local/bin/md-eval-22.pl
RUN chmod 755 /usr/local/bin/md-eval-22.pl
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_DISABLE_XET=1
WORKDIR /workspace
