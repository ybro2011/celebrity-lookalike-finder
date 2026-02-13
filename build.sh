#!/bin/bash
set -e

# Limit memory usage during build
export MAKEFLAGS="-j1"
export CMAKE_BUILD_PARALLEL_LEVEL=1

# Install dependencies one at a time to reduce memory pressure
pip install --no-cache-dir numpy>=1.24.0
pip install --no-cache-dir Pillow>=10.0.0
pip install --no-cache-dir Flask>=3.0.0
pip install --no-cache-dir gunicorn>=21.2.0
pip install --no-cache-dir requests>=2.31.0
pip install --no-cache-dir duckduckgo-search>=6.0.0

# Install opencv-python (pre-built wheel, faster)
pip install --no-cache-dir opencv-python-headless>=4.8.0

# Install mediapipe (pre-built)
pip install --no-cache-dir mediapipe==0.10.7

# Install dlib last (this is the memory killer)
# Use single thread to reduce memory usage
pip install --no-cache-dir --no-build-isolation dlib>=19.24.0

# Install face-recognition last (depends on dlib)
pip install --no-cache-dir face-recognition>=1.3.0
