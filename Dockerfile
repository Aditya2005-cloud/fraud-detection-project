FROM node:22-bookworm-slim AS frontend-builder

WORKDIR /frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build


FROM python:3.11-slim-bookworm AS runtime

WORKDIR /app

ARG VCS_REF=local
ARG BUILD_DATE=unknown

LABEL org.opencontainers.image.title="Fraud Detection Project" \
      org.opencontainers.image.description="Full-stack credit card fraud detection app with Flask, React, and ensemble machine learning models." \
      org.opencontainers.image.source="https://github.com/Aditya2005-cloud/fraud-detection-project" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY app.py ./app.py
COPY src ./src
COPY model ./model
COPY frontend/public ./frontend/public
COPY --from=frontend-builder /frontend/dist ./frontend/dist

EXPOSE 5000

# The health check keeps container platforms and Compose aware of app readiness.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:5000/health').read()"

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "app:app"]
