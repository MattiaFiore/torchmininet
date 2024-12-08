#!/bin/bash
# Define the directory path
TORCH_PATH="pytorch_download/"
TORCH_VISION_PATH='torchvision_download/'

# Check if the folder exists
if [ -d "$TORCH_PATH" ]; then
    echo "The folder $TORCH_PATH already exists!"
else
    pip download torch -d pytorch_download
fi

if [ -d "$TORCH_PATH" ]; then
    echo "The folder $TORCH_VISION_PATH already exists!"
else
    pip download torchvision -d torch_vision
fi
