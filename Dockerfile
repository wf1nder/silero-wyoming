ARG BASE_IMAGE=dr.wf1nder.me/hub/python:3.11-slim
FROM ${BASE_IMAGE}

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY silero_wyoming /app/silero_wyoming

WORKDIR /data

ENTRYPOINT ["python", "-m", "silero_wyoming"]
