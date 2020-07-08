from src.edge import Edges
from util.fedavg import FedAvg
import numpy as np
import torch
import os


class Cloud:
    def __init__(self, num_classes, edge_num, client_num, batch_size, device):
        self.edge = Edges(num_classes, edge_num, client_num, batch_size, device)
        self.model = self.edge.model

    def train(self, optimizer, scheduler, epochs, ratio1, ratio2, device):

        train_losses, train_accuracies = []
        test_losses, test_accuracies = []

        self.edge.model.train()
        global_vars = self.edge.model.state_dict()
        for epoch in range(1, epochs):
            edge_vars_sum = []
            random_edges = self.edge.choose_edges(ratio1)

            edge_loss, edge_acc = [], []

            for edge_id in random_edges:
                self.edge.set_global_vars(global_vars)
                train_acc, train_loss, current_edge_vars = self.edge.train_epoch(edge_id=edge_id, optimizer=optimizer, ratio2=ratio2, device=device)
                print(("[epoch {} ] edge_id:{}, Training Acc: {:.4f}, Loss: {:.4f}".format(
                    epoch, edge_id, train_acc, train_loss)))
                edge_vars_sum.append(current_edge_vars)

                edge_loss.append(train_loss)
                edge_acc.append(train_acc)
            
            global_vars = FedAvg(edge_vars_sum)
            self.edge.model.load_state_dict(global_vars)
            test_acc, test_loss = self.edge.run_test(device=device)
            print("[epoch {} ] Testing Acc: {:.4f}, Loss: {:.4f}".format(
                epoch, test_acc, test_loss))
            scheduler.step()

            # save loss
            train_losses.append(np.mean(edge_loss))
            train_accuracies.append(np.mean(edge_acc))
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)

        return train_losses, train_accuracies, test_losses, test_accuracies


