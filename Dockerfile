# Use a PyTorch base image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire project
COPY . .

# Install Python dependencies (including the package itself)
RUN pip install numpy torch

# Default command: run training
ENTRYPOINT ["python3", "-m", "src.training.trainer"]
CMD ["--episodes", "1000", "--batch_size", "32"]
