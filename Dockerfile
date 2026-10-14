# API image only (no training, notebooks or dashboard). Build after `python -m src.train` has written models/.
FROM python:3.12-slim

# LightGBM needs the OpenMP runtime, which the slim image leaves out
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY configs/ configs/
COPY src/ src/
COPY api/ api/
COPY models/ models/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
