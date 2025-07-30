# 1. Base image with llama.cpp lightweight executable and tools
FROM ghcr.io/ggml-org/llama.cpp:light

# 2. Install Python and pip
RUN apt-get update && apt-get install -y python3 python3-pip

# 3. Install Poetry for Python dependency management
RUN pip3 install poetry

# 4. Set working directory inside container
WORKDIR /insight-bridge

# 5. Copy your Python project files
COPY pyproject.toml poetry.lock /insight-bridge/

# 6. Install dependencies via Poetry (no dev dependencies)
RUN poetry config virtualenvs.create false
RUN poetry install --only main

COPY app /insight-bridge/app
COPY ingest /insight-bridge/ingest
COPY data /insight-bridge/data

# 7. Expose port if you run a server (optional)
EXPOSE 7860

# 8. Default command to run your app (adjust to your entrypoint)
CMD ["python3", "app/main.py"]
