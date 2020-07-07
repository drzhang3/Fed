import torch
import numpy as np
from collections import namedtuple
import math
from model import ResNet34
from data import DataSet
from client import Clients
import tqdm


class Edges:
    def __init__(self, num_classes, edge_num, client_num, batch_size, device):
        self.model = ResNet34(num_classes).to(device)
        self.bs = batch_size
        self.client_num = client_num
        self.edge_num = edge_num
        self.client_num_per_edge = client_num // edge_num
        self.client = Clients(num_classes, edge_num, client_num, batch_size, device)
        self.dataset = DataSet(edge_num, self.client_num_per_edge, batch_size)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train_epoch(self, edge_id, optimizer, device):
        edge_vars = self.client.get_client_vars()
        client_vars_sum = None
        random_clients = self.client.choose_clients(0.3)
        for client_id in random_clients:
            self.client.set_edge_vars(edge_vars)
            train_acc, train_loss = self.client.train_epoch(edge_id, client_id, optimizer=optimizer, device=device)
            current_client_vars = self.client.get_client_vars()
            if client_vars_sum is None:
                client_vars_sum = current_client_vars
            else:
                for cv, ccv in zip(client_vars_sum, current_client_vars):
                    cv = cv + ccv

        edge_vars = []
        for var in client_vars_sum:
            edge_vars.append(var / len(random_clients))

        return train_acc, train_loss

    def run_test(self, device):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.dataset.test):
                inputs, targets = inputs.to(device), targets.to(device)
                if inputs.size(-1) == 28:
                    inputs = inputs.view(inputs.size(0), -1)
                outputs = self.model(inputs)

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                loss = self.criterion(outputs, targets)
                test_loss += loss.item()

        accuracy = 100. * correct / total
        return accuracy, test_loss

    def _train_epoch(self, cid, optimizer, device):
        client_train_data_loader = self.dataset.train[cid]
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(client_train_data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            if inputs.size(-1) == 28:
                inputs = inputs.view(inputs.size(0), -1)
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            accuracy = 100. * correct / total
        return accuracy, train_loss

    def get_edge_vars(self):
        return self.model.parameters()

    def set_global_vars(self, edge_vars):
        all_vars = self.model.parameters()
        for variable, value in zip(all_vars, edge_vars):
            variable.data = value.data

    def choose_edges(self, ratio1):
        choose_num = math.floor(self.edge_num * ratio1)
        return np.random.permutation(self.edge_num)[:choose_num]
