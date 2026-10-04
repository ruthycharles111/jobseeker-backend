FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget unzip ca-certificates \
    chromium chromium-driver \
    fonts-liberation libasound2 libatk-bridge2.0-0 libatk1.0-0 libcups2 libdrm2 \
    libgbm1 libgtk-3-0 libnspr4 libnss3 libx11-xcb1 libxcomposite1 libxdamage1 \
    libxrandr2 xdg-utils \
    && rm -rf /var/lib/apt/lists/*

# PocketBase
RUN wget -q -O /tmp/pb.zip https://github.com/pocketbase/pocketbase/releases/download/v0.22.21/pocketbase_0.22.21_linux_amd64.zip \
    && unzip /tmp/pb.zip -d /app && rm /tmp/pb.zip && chmod +x /app/pocketbase

# Litestream
RUN wget -q https://github.com/benbjohnson/litestream/releases/download/v0.3.13/litestream-v0.3.13-linux-amd64.tar.gz \
    && tar -xzf litestream-v0.3.13-linux-amd64.tar.gz -C /app && rm litestream-v0.3.13-linux-amd64.tar.gz && chmod +x /app/litestream

# Python virtual environment for worker
RUN python -m venv /app/worker-venv

# Copy requirements and install (this now includes serpapi, google-search-results, etc.)
COPY worker/requirements.txt /app/worker-requirements.txt
RUN /app/worker-venv/bin/pip install --no-cache-dir -r /app/worker-requirements.txt

# Copy all remaining files
COPY run.sh /app/run.sh
COPY litestream.yml /app/litestream.yml
COPY worker/worker.py /app/worker.py
COPY worker/supervisor.py /app/supervisor.py
COPY worker/source_probe.py /app/source_probe.py

RUN chmod +x /app/run.sh && mkdir -p /app/pb_data

EXPOSE 8090
CMD ["/app/run.sh"]
