FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY pyproject.toml .
COPY backend/ backend/
COPY configs/ configs/

# Install dependencies
RUN uv sync --no-dev

# Create required directories
RUN mkdir -p models data/episodes runs

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
