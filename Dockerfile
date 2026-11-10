FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY pyproject.toml .
COPY finmon ./finmon
RUN pip install --upgrade pip && pip install .

COPY evals ./evals
COPY data/raw ./data/raw
COPY data/synthetic ./data/synthetic

# Дані синтетичні, індекс будується під час збирання образу (модель ембедингів завантажується тут же)
RUN python -m finmon.rag.index minilm structural_short

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "finmon.api:app", "--host", "0.0.0.0", "--port", "8000"]
