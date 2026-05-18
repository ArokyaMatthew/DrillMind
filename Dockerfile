FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (for caching)
COPY pyproject.toml .
RUN pip install --no-cache-dir -e . 2>/dev/null || \
    (pip install --no-cache-dir \
        pandas>=2.0 numpy>=1.24 openpyxl>=3.1 pyyaml>=6.0 \
        pydantic>=2.0 loguru>=0.7 websockets>=12.0 \
        fastapi>=0.110 uvicorn>=0.29 \
        torch --index-url https://download.pytorch.org/whl/cpu \
        scikit-learn \
        chromadb>=0.5 sentence-transformers>=3.0 datasets>=3.0)

# Copy source code
COPY . .
RUN pip install --no-cache-dir -e .

# Expose API port
EXPOSE 8000

# Default: load 50K rows for reasonable startup time
ENV DRILLMIND_MAX_ROWS=50000

CMD ["python", "-m", "uvicorn", "drillmind.api.server:app", \
     "--host", "0.0.0.0", "--port", "8000"]
