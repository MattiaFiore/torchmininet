#!/bin/bash

# Install PyTorch from the local directory
cd pytorch_download
pip install --no-index --find-links . torch
cd ..
cd torchvision_download
# Install TorchVision from the local directory
pip install --no-index --find-links . torchvision
