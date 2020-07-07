import torch
import torch.optim as optim
from tqdm import tqdm
import argparse
import numpy as np
import random

from client import Clients
from edge import Edges


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # 并行gpu
    torch.backends.cudnn.deterministic = True  # cpu/gpu结果一致
    torch.backends.cudnn.benchmark = True  # 训练集变化不大时使训练加速


def get_parse():
    # parser for hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--edge_num', type=int, default=10)
    parser.add_argument('--client_num', type=int, default=100)
    parser.add_argument('--ratio1', type=float, default=0.3, help='The ratio of chosen edges')
    parser.add_argument('--ratio2', type=float, default=0.8, help='The ratio of chosen client per edge')
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


def run_global_test(edge, global_vars):
    edge.set_global_vars(global_vars)
    acc, loss = edge.run_test(device=device)
    return acc, loss


if __name__ == '__main__':
    args = get_parse()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    edge = Edges(args.num_classes, args.edge_num, args.client_num, args.bs, device)
    global_vars = edge.get_edge_vars()
    optimizer = create_optimizer(args, edge.model.parameters())
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 300], gamma=0.1)
    for epoch in range(1, args.epochs):
        edge_vars_sum = None
        random_edges = edge.choose_edges(args.ratio1)
        for edge_id in tqdm(random_edges, ascii=True):
            edge.set_global_vars(global_vars)
            train_acc, train_loss = edge.train_epoch(edge_id=edge_id, optimizer=optimizer, device=device)
            print(("[epoch {} ] edge_id:{}, Training Acc: {:.4f}, Loss: {:.4f}".format(
                epoch, edge_id, train_acc, train_loss)))
            current_edge_vars = edge.get_edge_vars()
            if edge_vars_sum is None:
                edge_vars_sum = current_edge_vars
            else:
                for cv, ccv in zip(edge_vars_sum, current_edge_vars):
                    cv = cv + ccv

        global_vars = []
        for var in edge_vars_sum:
            global_vars.append(var / len(random_edges))

        test_acc, test_loss = run_global_test(edge, global_vars)
        print("[epoch {} ] Testing Acc: {:.4f}, Loss: {:.4f}".format(
            epoch, test_acc, test_loss))

        scheduler.step()
