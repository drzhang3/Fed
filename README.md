# Federated Learning

We simulate the Cloud-Edge-Client FL framework.

## Requirements

python==3.7

pytorch==1.4.0

## Run

Federated learning with ResNet is produced by:
> python [main.py](main.py)

See the arguments in [main.py](main.py). 

For example:
> python main.py --edge_num=10 --client_num=100 --ratio1=0.5 --ratio2=0.3 --optim=adam --lr=0.001 --bs=128

Note: We only consider IID setting on CIFAR10.






