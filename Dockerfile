FROM python:3.11.9

# Environment settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/home/user/.cache/huggingface
ENV TRANSFORMERS_CACHE=/home/user/.cache/huggingface/transformers
ENV DIFFUSERS_CACHE=/home/user/.cache/huggingface/diffusers

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Create a non-privileged user for HF Spaces
RUN useradd -m -u 1000 user
USER user

# Install Python dependencies
# Copy root requirements and remove Windows-specific packages
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy application code
COPY --chown=user . .

EXPOSE 7860

# Run the startup script
CMD ["startup.sh"]