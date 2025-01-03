FROM nvcr.io/nvidia/pytorch:24.12-py3

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.8.5 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv"

ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    curl build-essential

RUN curl -sSL https://install.python-poetry.org | python3 - --version $POETRY_VERSION

WORKDIR $PYSETUP_PATH
COPY poetry.lock pyproject.toml ./

RUN poetry env use system

RUN --mount=type=cache,target=/root/.cache/pypoetry poetry install

COPY scripts scripts
COPY utils utils
COPY infer.py train.py .

COPY conf conf

ENTRYPOINT ["poetry", "run", "python"]
