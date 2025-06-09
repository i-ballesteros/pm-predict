FROM us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-6:latest

WORKDIR /app

COPY pyproject.toml .

COPY src/ src/

RUN pip install --upgrade pip \
    && pip install .

ENTRYPOINT ["python", "-m", "src.train"]
