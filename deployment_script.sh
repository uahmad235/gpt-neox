#!/usr/bin/env bash -l
set -euo pipefail

sudo apt-get update && sudo apt-get install -y nano
#set -euo pipefail
echo $(which pip)
eval "$(conda shell.bash hook)"
conda create -y --name gpt_neox python=3.8
conda activate gpt_neox
echo $(which pip)

echo "---- Cloning GPT-NEOX repo -----"
git clone https://github.com/uahmad235/gpt-neox.git

echo "--- Downloading model weights --- "
(cd gpt-neox && wget --cut-dirs=5 -nH -r --no-parent --reject "index.html*" https://the-eye.eu/public/AI/models/GPT-NeoX-20B/slim_weights/ -P 20B_checkpoints)

echo "---- Installing ubuntu dependencies -----"
sudo apt-get install -y libopenmpi-dev

echo "---- Installing code dependencies -----"

pip install Cython \
    && pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 \
    && pip install -r gpt-neox/requirements/requirements.txt \
    && pip install protobuf==3.20.* \
    && pip install fastapi==0.72.0 \
    && pip install uvicorn==0.16.0

python gpt-neox/megatron/fused_kernels/setup.py install

(cd gpt-neox && python app.py)
