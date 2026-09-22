FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.docker.txt .
RUN pip install --no-cache-dir -r requirements.docker.txt

COPY accounts.py alerts.py app.py faces.py main.py pipeline.py session.py zones.py ./
COPY static ./static

ENV DATA_DIR=/data
ENV ALERTS_DIR=/data/alerts
RUN mkdir -p /data/alerts
VOLUME /data
EXPOSE 8000

CMD ["python", "app.py"]
