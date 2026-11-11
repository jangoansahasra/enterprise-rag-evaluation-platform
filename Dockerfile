ARG PYTHON_VERSION=3.13

FROM python:${PYTHON_VERSION}-slim AS runtime-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN groupadd --gid 10001 ragapp \
    && useradd \
        --uid 10001 \
        --gid ragapp \
        --create-home \
        --shell /usr/sbin/nologin \
        ragapp


FROM runtime-base AS dashboard

RUN python -m pip install --upgrade pip \
    && python -m pip install \
        pandas==3.0.6 \
        plotly==7.1.0 \
        streamlit==1.64.0

COPY --chown=ragapp:ragapp .streamlit ./.streamlit
COPY --chown=ragapp:ragapp dashboard ./dashboard
COPY --chown=ragapp:ragapp data ./data
COPY --chown=ragapp:ragapp config ./config
COPY --chown=ragapp:ragapp artifacts/baselines ./artifacts/baselines

USER ragapp

EXPOSE 8501

HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=30s \
    --retries=3 \
    CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3)"

CMD ["python", "-m", "streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]


FROM runtime-base AS pipeline

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --upgrade pip \
    && python -m pip install ".[dev]"

COPY --chown=ragapp:ragapp . .

USER ragapp

CMD ["python", "scripts/validate_data.py"]