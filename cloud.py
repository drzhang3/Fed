from edge import Edges
import tqdm


class Cloud:
    def __init__(self, num_classes, edge_num, client_num, batch_size, device):
        self.edge = Edges(num_classes, edge_num, client_num, batch_size, device)
        self.model = self.edge.model

    def train(self, optimizer, scheduler, epochs, ratio1, ratio2, device):
        global_vars = self.edge.get_edge_vars()
        for epoch in range(1, epochs):
            edge_vars_sum = None
            random_edges = self.edge.choose_edges(ratio1)
            # for edge_id in tqdm(random_edges, ascii=True):
            for edge_id in random_edges:
                self.edge.set_global_vars(global_vars)
                train_acc, train_loss = self.edge.train_epoch(edge_id=edge_id, optimizer=optimizer, ratio2=ratio2,
                                                         device=device)
                print(("[epoch {} ] edge_id:{}, Training Acc: {:.4f}, Loss: {:.4f}".format(
                    epoch, edge_id, train_acc, train_loss)))
                # current_edge_vars = edge.get_edge_vars()
                current_edge_vars = self.edge.edge_vars
                if edge_vars_sum is None:
                    edge_vars_sum = current_edge_vars
                else:
                    for cv, ccv in zip(edge_vars_sum, current_edge_vars):
                        cv = cv + ccv

            global_vars = []
            for var in edge_vars_sum:
                global_vars.append(var / len(random_edges))

            self.edge.set_vars(global_vars)
            test_acc, test_loss = self.edge.run_test(device=device)
            print("[epoch {} ] Testing Acc: {:.4f}, Loss: {:.4f}".format(
                epoch, test_acc, test_loss))
            scheduler.step()


