# Distributed Training on Mininet

In this repository torch distributed training is implemented inside mininet hosts. 

- [P4 switch](./p4src/connecting.p4): This switch is used just to forward packets from and to the parameter server. The forwarding is implemented manually inside a [txt file](./s1-commands.txt). 
- [classifier.py](./classifier.py) implements the training of a neural network on the MNIST dataset 
- [start_network.sh](start_network.sh) is responsible for starting the network and making sure that torch and torchvision are available on the hosts. 

Currently the code can only train the model on the hosts not in a distributed fashion. 