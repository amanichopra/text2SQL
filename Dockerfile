FROM gcr.io/deeplearning-platform-release/pytorch-gpu.1-13.py310:latest

# Configure Poetry
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.1.4  \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv" 

# env variables
ENV SPIDER_BUCKET="spider-dataset"
ENV COSQL_BUCKET="cosql-dataset"

ENV MODEL_CHECKPOINT_BUCKET="mdl-checkpoints"
ENV T5_WIKISQL_COSQL_CHECKPOINT_PATH="t5_wikisql_cosql"

WORKDIR /app

COPY . ./

# install requirements
RUN pip install -r requirements.txt