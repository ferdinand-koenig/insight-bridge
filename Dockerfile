########################################
# Builder stage: installs Python deps in virtualenv
########################################
FROM ghcr.io/ggml-org/llama.cpp:light AS builder

ARG PYTHON_VERSION=3.10

# 1. Install Python and pip
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-venv \
        python${PYTHON_VERSION}-distutils \
        python3-pip

# 2. Install pinned Poetry version
RUN pip3 install poetry==2.1.2

# 3. Configure Poetry environment (create in-project virtualenv)
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /insight-bridge

# 4. Copy dependency files and dummy README for Poetry sanity
COPY pyproject.toml poetry.lock ./
COPY README.md .

# 5. Install dependencies (no dev), no local package install yet
RUN poetry install --only main --no-root && rm -rf $POETRY_CACHE_DIR

# 6. Copy actual app code
COPY app ./app
COPY ingest ./ingest
COPY data ./data

# 7. Install your local package (now that source is present)
RUN poetry install --without dev

########################################
# Runtime stage: minimal runtime with llama.cpp + Python
########################################
FROM ghcr.io/ggml-org/llama.cpp:light AS runtime

ARG PYTHON_VERSION=3.10

# 8. Install matching Python version
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-venv \
        python${PYTHON_VERSION}-distutils \
        python3-pip

# 9. Set up environment for virtualenv and add path for llama lib
ENV VIRTUAL_ENV=/insight-bridge/.venv \
    PATH="/insight-bridge/.venv/bin:$PATH" \
    LD_LIBRARY_PATH=/app:${LD_LIBRARY_PATH:-}


WORKDIR /insight-bridge

# 10. Copy virtualenv from builder
COPY --from=builder /insight-bridge/.venv /insight-bridge/.venv

# 11. Copy project files (again — but clean, no dev tooling)
COPY app ./app
COPY ingest ./ingest
COPY data ./data
COPY README.md .

# 12. Expose Gradio or API port
EXPOSE 7860

# 13. Default command to run your Gradio app
CMD ["python3", "-m", "app.main"]
