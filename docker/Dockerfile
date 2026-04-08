FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y \
    python3.11 python3-pip python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Copy application code
COPY config/ config/
COPY src/ src/

# Copy model artifacts (if available)
COPY models/ models/ 2>/dev/null || true

# Switch to non-root user
USER appuser

# Expose ports
EXPOSE 8000 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Default: start the model serving API
CMD ["python3", "-m", "uvicorn", "src.serving.model_server:app", \
     "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
