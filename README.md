# Distributed Training on Mininet

In this repository torch distributed training is implemented inside mininet hosts. 

## Running the repo
The code will start the mininet topology and install the torch library in each worker
```
bash start_network.sh 
```
For each worker run the bash script referred to their name after opening the terminal in xterm 

## Notes

- [P4 switch](./p4src/connecting.p4): This switch is used just to forward packets from and to the parameter server. The forwarding is implemented manually inside a [txt file](./s1-commands.txt). 
- [classifier.py](./classifier.py) implements the training of a neural network on the MNIST dataset 
- [start_network.sh](start_network.sh) is responsible for starting the network and making sure that torch and torchvision are available on the hosts. 

