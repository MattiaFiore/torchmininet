#!/bin/bash

# tcpdump -w ps0.pcap & 

export RANK=2
export WORLD_SIZE=3 
export MASTER_ADDR=10.0.0.2 
export MASTER_PORT=29500 
export TP_SOCKET_IFNAME=h1-eth0
export GLOO_SOCKET_IFNAME=h1-eth0

python distributed.py

