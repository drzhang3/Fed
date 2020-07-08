import torch
import torch.optim as optim
import argparse
import numpy as np
import random
import os
from src.cloud import Cloud
import matplotlib.pyplot as plt


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_parse():
    # parser for hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fixed_seed', type=bool, default=False)
    parser.add_argument('--edge_num', type=int, default=1)
    parser.add_argument('--client_num', type=int, default=1)
    parser.add_argument('--ratio1', type=float, default=1, help='The ratio of chosen edges')
    parser.add_argument('--ratio2', type=float, default=1, help='The ratio of chosen client per edge')
    parser.add_argument('--optim', default='adam', type=str, help='optimizer')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum term')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam coefficients beta_1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam coefficients beta_2')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='weight decay for optimizers')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--num_classes', type=int, default=10, help='cifar10')
    args = parser.parse_args()
    return args


def create_optimizer(args, model_params):
    if args.optim == 'sgd':
        return optim.SGD(model_params, args.lr, momentum=args.momentum,
                         weight_decay=args.weight_decay)
    elif args.optim == 'adagrad':
        return optim.Adagrad(model_params, args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        return optim.Adam(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay)
    elif args.optim == 'amsgrad':
        return optim.Adam(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay, amsgrad=True)
    else:
        raise ValueError('unknown optimizer')


def plot(curve, name):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Training loss')
    plt.plot(curve)
    plt.legend()
    plt.savefig('figure/{}.png'.format(name))


if __name__ == '__main__':

    args = get_parse()
    if args.fixed_seed:
        setup_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cloud = Cloud(args.num_classes, args.edge_num, args.client_num, args.bs, device)
    optimizer = create_optimizer(args, cloud.model.parameters())
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 300], gamma=0.1)
    train_loss, train_acc, test_loss, test_acc = cloud.train(optimizer, scheduler, args.epochs, args.ratio1,
                                                             args.ratio1, device)

    if not os.path.isdir('curve'):
        os.mkdir('curve')
    name = 'fed-edge{}-client{}_{}_C{}'.format(args.edge_num, args.client_num, args.ratio1, args.ratio2)
    torch.save({'train_loss': train_loss, 'train_accuracy': train_acc,
                'test_loss': test_loss, 'test_accuracy': test_acc}, name)

    plot(train_loss, name)
