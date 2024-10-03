FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libasound2-dev \
    portaudio19-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

WORKDIR /app

RUN pip3 install --no-cache-dir -r requirements.txt

USER 1001
EXPOSE 8080
CMD ["python3", "/app/transcribe.py"]