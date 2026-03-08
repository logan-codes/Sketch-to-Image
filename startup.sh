#!/bin/bash
echo "=== Starting Sketch to Image App ==="
echo "Python version:"
python --version
echo "=== Environment Info ==="
echo "Current directory: $(pwd)"
echo "Available disk space:"
df -h
echo "=== Memory Info ==="
free -h
echo "=== GPU Info ==="
nvidia-smi || echo "No GPU available"
echo "=== Starting Application ==="
python app.py
