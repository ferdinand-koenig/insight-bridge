########################################
# Builder Worker stage: installs Python deps in virtualenv
########################################
FROM ghcr.io/ggml-org/llama.cpp:light AS builder

ARG PYTHON_VERSION=3.10

# 1. Install Python and pip
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-venv \
        python${PYTHON_VERSION}-distutils \
        python3-pip \
        build-essential \
        cmake \
        python3-dev \
        libopenblas-dev \
        pkg-config \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Install pinned Poetry version
RUN pip3 install poetry==2.1.2

# 3. Configure Poetry environment (create in-project virtualenv) and enable OpenBLAS for llama-cpp-python
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_NO_BINARY="llama-cpp-python" \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    CMAKE_ARGS="-DLLAMA_CUBLAS=off -DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS -DLLAMA_NATIVE=ON -DLLAMA_BUILD_BACKEND=ON"


WORKDIR /insight-bridge

# 4. Copy dependency files and dummy README for Poetry sanity
COPY pyproject.toml poetry.lock ./
COPY README.md .
COPY install.py .

# 5. Install all dependencies, forcing llama-cpp-python to build from source
# RUN poetry run pip install --no-binary llama-cpp-python llama-cpp-python && \
RUN python3 install.py inference && \
    rm -rf $POETRY_CACHE_DIR

COPY config.yaml ./
RUN . .venv/bin/activate && python -c "import yaml; \
    from langchain_huggingface import HuggingFaceEmbeddings; \
    model=yaml.safe_load(open('config.yaml')).get('embedding_model_name', 'sentence-transformers/sci-base'); \
    HuggingFaceEmbeddings(model_name=model)"


########################################
# Runtime Worker stage: minimal runtime with llama.cpp + Python
########################################
FROM ghcr.io/ggml-org/llama.cpp:light AS runtime

ARG PYTHON_VERSION=3.10

# 8. Install matching Python version
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-venv \
        python${PYTHON_VERSION}-distutils \
        python3-pip \
        libopenblas-base

# 9. Set up environment for virtualenv and add path for llama lib
ENV VIRTUAL_ENV=/insight-bridge/.venv \
    PATH="/insight-bridge/.venv/bin:$PATH" \
    LD_LIBRARY_PATH=/insight-bridge/.venv/lib/python3.10/site-packages/llama_cpp:/app:${LD_LIBRARY_PATH:-}



WORKDIR /insight-bridge

# 10. Copy virtualenv from builder
COPY --from=builder /insight-bridge/.venv /insight-bridge/.venv
# Copy pre-downloaded Hugging Face cache
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

# 11. Copy project files (again — but clean, no dev tooling)
COPY app/__init__.py app/logger.py ./app/
COPY app/inference ./app/inference
#COPY ingest ./ingest
COPY data/faiss_index ./data/
COPY data/faiss_metadata.pkl ./data/
COPY README.md .
COPY config.yaml prompt_template.yaml ./

## Copy backend .so files into /insight-bridge
#RUN cp /app/libllama.so /insight-bridge/ && \
#    cp /app/libggml-*.so /insight-bridge/

# Link libllama.so
RUN ln -s /app/libllama.so /insight-bridge/libllama.so && \
    for file in /app/libggml-*.so; do \
        ln -s "$file" /insight-bridge/"$(basename "$file")"; \
    done



# 12. Expose FastAPI
EXPOSE 8000

# Remove the current entrypoint which was set to /app/llama-cli by base image
ENTRYPOINT []
# 13. Default command to run worker app
CMD ["python3", "-m", "app.inference.worker_server"]


########################################
# Server stage
########################################
FROM python:3.10-slim AS server

#ENV POETRY_CACHE_DIR=/tmp/poetry_cache \
ENV POETRY_VIRTUALENVS_CREATE=false

WORKDIR /insight-bridge

# Minimal dependencies
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
    apt-get clean && pip3 install --upgrade pip && rm -rf /var/lib/apt/lists/*


COPY README.md ./
COPY pyproject.toml poetry.lock ./
#COPY install.py .

# 5. Install all dependencies, forcing llama-cpp-python to build from source
RUN pip3 install --no-cache-dir poetry==2.1.2
RUN poetry self add poetry-plugin-export \
    && poetry export \
      -f requirements.txt \
      --output requirements.txt \
      --without-hashes \
      --with server \
    && pip3 uninstall -y poetry \
    && rm -f pyproject.toml poetry.lock

# Copy only what's necessary for Gradio server
COPY app/__init__.py app/logger.py ./app/
COPY app/server ./app/server
COPY app/llm_worker_provisioners/ ./app/llm_worker_provisioners/
COPY assets ./assets
COPY config.yaml ./
COPY app/tests ./app/tests
# TODO remove the test directory

# Make the script executable
RUN chmod +x ./app/server/entrypoint.sh

# Set the entrypoint to the script
ENTRYPOINT ["./app/server/entrypoint.sh"]
CMD ["python3", "-m", "app.server.main"]
