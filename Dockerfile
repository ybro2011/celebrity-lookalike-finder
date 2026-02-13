FROM mambaorg/micromamba:latest

# Switch to root to install minimal system libs for OpenCV
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up work directory and permissions
WORKDIR /app
RUN chown $MAMBA_USER:$MAMBA_USER /app

# Switch to micromamba user
USER $MAMBA_USER

# Install Python, pip, dlib, opencv, and gunicorn via micromamba
# This bypasses all heavy compilation steps
RUN micromamba install -y -n base -c conda-forge \
    python=3.11 \
    pip \
    dlib \
    opencv \
    gunicorn \
    && micromamba clean --all --yes

# Copy requirements
COPY --chown=$MAMBA_USER:$MAMBA_USER requirements.txt .

# Set environment path to include the micromamba base environment
ENV PATH="/opt/conda/bin:$PATH"

# Install the remaining lighter packages via pip
# We remove the heavy ones that we already installed via mamba
RUN sed -i '/dlib/d' requirements.txt && \
    sed -i '/opencv/d' requirements.txt && \
    sed -i '/gunicorn/d' requirements.txt && \
    python -m pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY --chown=$MAMBA_USER:$MAMBA_USER . .

# Ensure the celebs directory exists and is writable
RUN mkdir -p /app/celebs && chown -R $MAMBA_USER:$MAMBA_USER /app/celebs

# Hugging Face default port
EXPOSE 7860

# We use the $PORT env var if provided, otherwise default to 7860
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT:-7860} --workers 1 --timeout 120 --access-logfile - --error-logfile -"]
