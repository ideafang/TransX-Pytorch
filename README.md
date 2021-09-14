# TransX-PyTorch

Knowledge Graph Completion by Cross Attention in Translational Model

## Environment

This code is based on the [OpenKE-Pytorch](https://github.com/thunlp/OpenKE) project.

We evaluate TransX on a Ubuntu 16.04 server with 128G memory an RTX 3090.

## Installation
1. Install Pytorch
2. Compile C++ files
   ```shell
   cd openke
   bash make.sh
   ```
3. Quick Start
   ```shell
   cd ..
   python train_TransX_Dataset.py
   ```
