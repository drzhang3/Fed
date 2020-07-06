import torch
import torch.optim as optim
from tqdm import tqdm
import argparse
import numpy as np
import random

from client import Clients


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
    parser.add_argument('--num', type=int, default=100)
    parser.add_argument('--optim', default='adam', type=str, help='optimizer')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum term')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam coefficients beta_1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam coefficients beta_2')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='weight decay for optimizers')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=150)
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


def buildClients(arg, device):
    num_classes = 10

    return Clients(num_classes, args.bs, args.num, device)


def run_global_test(client, global_vars):
    client.set_global_vars(global_vars)
    acc, loss = client.run_test(device=device)
    return acc, loss
    # print("[epoch {}, {} inst] Testing ACC: {:.4f}, Loss: {:.4f}".format(
    #     ep + 1, test_num, acc, loss))


if __name__ == '__main__':
    args = get_parse()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    client = buildClients(args, device)
    global_vars = client.get_client_vars()
    optimizer = create_optimizer(args, client.model.parameters())
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.1)
    for epoch in range(1, args.epochs):
        client_vars_sum = None
        random_clients = client.choose_clients()
        for client_id in tqdm(random_clients, ascii=True):
            client.set_global_vars(global_vars)
            client.train_epoch(cid=client_id, optimizer=optimizer, device=device)
            current_client_vars = client.get_client_vars()
            if client_vars_sum is None:
                client_vars_sum = current_client_vars
            else:
                for cv, ccv in zip(client_vars_sum, current_client_vars):
                    cv = cv + ccv

        global_vars = []
        for var in client_vars_sum:
            global_vars.append(var / len(random_clients))

        acc, loss = run_global_test(client, global_vars)
        print("[epoch {} ] Testing ACC: {:.4f}, Loss: {:.4f}".format(
            epoch + 1, acc, loss))

        scheduler.step()
    run_global_test(client, global_vars)
