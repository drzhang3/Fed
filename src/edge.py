import numpy as np
import math
from src.client import Clients
from util.fedavg import FedAvg


class Edges:
    def __init__(self, num_classes, edge_num, client_num, batch_size, device):
        self.bs = batch_size
        self.client_num = client_num
        self.edge_num = edge_num
        self.client_num_per_edge = client_num // edge_num
        self.client = Clients(num_classes, edge_num, client_num, batch_size, device)
        self.model = self.client.model

    def train_epoch(self, edge_id, optimizer, ratio2, device):
        
        edge_vars = self.edge_vars
        client_vars_sum = []
        random_clients = self.client.choose_clients(ratio2)
        for client_id in random_clients:
            self.client.set_edge_vars(edge_vars)
            train_acc, train_loss, current_client_vars = self.client.train_epoch(edge_id, client_id, optimizer=optimizer, device=device)
            # current_client_vars = self.client.get_client_vars()
            client_vars_sum.append(current_client_vars)

        edge_vars = FedAvg(client_vars_sum)
        self.edge_vars = edge_vars
        return train_acc, train_loss, edge_vars

    def run_test(self, device):
        accuracy, test_loss = self.client.run_test(device)
        return accuracy, test_loss

    def set_global_vars(self, edge_vars):
        self.edge_vars = edge_vars

    def choose_edges(self, ratio1):
        choose_num = math.floor(self.edge_num * ratio1)
        return np.random.permutation(self.edge_num)[:choose_num]
