FROM mambaorg/micromamba:latest

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    libxcb-xinerama0 \
    libxcb-shm0 \
    libxcb-randr0 \
    libxcb-render0 \
    libxcb-render-util0 \
    libxcb-xfixes0 \
    libxcb-xkb1 \
    libxcb-image0 \
    libxcb-icccm4 \
    libxcb-keysyms1 \
    libxcb-glx0 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN chown $MAMBA_USER:$MAMBA_USER /app

USER $MAMBA_USER

RUN micromamba install -y -n base -c conda-forge \
    python=3.11 \
    pip \
    setuptools \
    dlib \
    opencv \
    face-recognition \
    gunicorn \
    && micromamba clean --all --yes

COPY --chown=$MAMBA_USER:$MAMBA_USER requirements.txt .

ENV PATH="/opt/conda/bin:$PATH"
ENV PORT=8080

RUN sed -i '/dlib/d' requirements.txt && \
    sed -i '/opencv/d' requirements.txt && \
    sed -i '/gunicorn/d' requirements.txt && \
    sed -i '/face-recognition/d' requirements.txt && \
    /opt/conda/bin/python -m pip install --no-cache-dir -r requirements.txt

COPY --chown=$MAMBA_USER:$MAMBA_USER . .

RUN /opt/conda/bin/python -m pip install --no-cache-dir "setuptools<66" && \
    /opt/conda/bin/python -m pip install --no-cache-dir git+https://github.com/ageitgey/face_recognition_models && \
    /opt/conda/bin/python -c "import face_recognition_models; print('face_recognition_models installed successfully')" && \
    mkdir -p /app/celebs && chown -R $MAMBA_USER:$MAMBA_USER /app/celebs

EXPOSE 8080

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-"]
