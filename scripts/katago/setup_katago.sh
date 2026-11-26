#!/bin/bash
# Installs KataGo (CUDA build) + nets into /content/katago. Prints versions/URLs. Idempotent.
set -e
VER=v1.18.2
ZIP=katago-$VER-cuda12.8-cudnn9.8.0-linux-x64.zip
MAIN=kata1-b18c384nbt-s9996604416-d4316597426.bin.gz
HUMAN=b18c384nbt-humanv0.bin.gz
mkdir -p /content/katago && cd /content/katago
[ -f katago ] || { wget -q https://github.com/lightvector/KataGo/releases/download/$VER/$ZIP && unzip -oq $ZIP && chmod +x katago; }
[ -f $MAIN ] || wget -q https://media.katagotraining.org/uploaded/networks/models/kata1/$MAIN
[ -f $HUMAN ] || wget -q https://media.katagotraining.org/uploaded/networks/models_extra/$HUMAN
ls /content/cu12/nvidia >/dev/null 2>&1 || pip install -q --upgrade --target /content/cu12 nvidia-cublas-cu12==12.8.4.1 nvidia-cuda-runtime-cu12==12.8.90 nvidia-cudnn-cu12==9.8.0.87 nvidia-cuda-nvrtc-cu12==12.8.93
ls -la /content/katago | head -20
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
ldd ./katago | grep -i "not found" || echo "ldd ok"
echo "ZIP_URL=https://github.com/lightvector/KataGo/releases/download/$VER/$ZIP"
