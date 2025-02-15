#INSTALL ON EACH WORKER TORCH AND TORCHVISION
HOSTS=('h0' 'h1' 'ps0')
THREADS=('0' '4', '8')


for i in "${HOSTS[@]}"; do

  HOST=${HOSTS[$i]}
  THREAD_RANGE=${THREADS[$i]}

  echo "INSTALLING TORCH AND TORCH VISION FOR WORKER $HOST"
  #Finding the PID of heach host 
  PID=$(ps aux | grep $HOST | grep 'bash' | awk '{print $2}')
  echo "$PID"

  # Set CPU affinity for the process using taskset
  #echo "Setting CPU affinity for $HOST (PID: $PID) to cores $THREAD_RANGE"
  #taskset -cp $THREAD_RANGE $PID

  #Execute the worker code 
  echo "dioporco"
  sudo mnexec -a $PID bash bash_scripts/install_torch_worker.sh
  sudo mnexec -a $PID sudo ufw disable

done
