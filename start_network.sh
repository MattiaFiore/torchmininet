#!/bin/bash

#VERIFY IF THE TORCH AND TORCH VISION LIBRARIES ARE INSTALLED 
sudo bash bash_scripts/install_torch_folder.sh

#START THE NETWORK 
sudo python3 network.py
