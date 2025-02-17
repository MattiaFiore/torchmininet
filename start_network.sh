#!/bin/bash

# START THE NETWORK 
xterm -hold -e sudo python3 network.py &
# sleep 7

# INSTALL ON EACH WORKER TORCH AND TORCHVISION
HOSTS=('h0' 'h1' 'ps0')
THREADS=('0-1' '2-3' '4-5')

for i in "${!HOSTS[@]}"; do 
  HOST=${HOSTS[$i]}
  THREAD_RANGE=${THREADS[$i]}
  
  # Wait until the PID is found for the host
  while true; do
    PID=$(pgrep -f "mininet:$HOST")
    if [[ -n "$PID" ]]; then
      break
    fi
    sleep 1
  done

  # Disable the firewall and open xterm for the hostudo mnexec -a $PID xterm -e "bash start_$HOST.sh" &
  sudo mnexec -a $PID sudo ufw disable
  sudo mnexec -a $PID xterm -e "bash start_$HOST.sh" &

done
