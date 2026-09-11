FROM swr.cn-south-1.myhuaweicloud.com/ascendhub/torch-npu@sha256:4a46a45ea932c72a0d57f0de9aebb0d13021329d778c6dd0810ce7c0da60a4f6
USER root
RUN mkdir -p /tmp && chmod 1777 /tmp && yum install -y gcc gcc-c++ git cmake make libsndfile && yum clean all
RUN python -m pip install --no-cache-dir 'numpy<2' setuptools wheel cmake ninja \
    'lhotse==1.23.0' 'sentencepiece==0.2.0' tensorboard kaldifst kaldialign \
    pypinyin dill typeguard regex PyYAML pydantic pydantic-settings openai anthropic \
    google-genai 'mcp==1.26.0' flask python-dotenv jsonschema requests pillow rich
ARG K2_COMMIT=ec31d2c96e04665eadb276ebcfd436892f74b0aa
RUN git clone https://github.com/k2-fsa/k2.git /opt/k2-src \
    && cd /opt/k2-src && git checkout "$K2_COMMIT" \
    && TORCH_DEVICE_BACKEND_AUTOLOAD=0 K2_CMAKE_ARGS='-DK2_WITH_CUDA=OFF -DK2_WITH_TESTS=OFF' K2_MAKE_ARGS='-j4' \
       python -m pip install --no-build-isolation .
RUN TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pip install --no-cache-dir 'torchaudio==2.10.0' 'lhotse==1.33.0'
RUN git -c http.version=HTTP/1.1 clone --depth 1 --branch 2.5 https://github.com/burrmill/sph2pipe.git /opt/sph2pipe-src \
    && cd /opt/sph2pipe-src && git checkout 62db964f20cc2d180a18514e1cee10deb36af3a5 \
    && make && cp sph2pipe /usr/local/bin/sph2pipe
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /workspace
