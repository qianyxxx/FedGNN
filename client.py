# federated/client.py
import torch
from copy import deepcopy

class Client:
    def __init__(self, client_id, model, dataloader, optimizer_cls, loss_fn, device='cpu'):
        self.client_id = client_id
        self.model = deepcopy(model).to(device)
        self.dataloader = dataloader
        self.optimizer = optimizer_cls(self.model.parameters())
        self.loss_fn = loss_fn
        self.device = device

    # 现在 `train` 方法正确地成为了 `Client` 类的一部分
    def train(self, epochs=1):
        self.model.train()
        for epoch in range(epochs):
            for batch in self.dataloader:
                batch.to(self.device)
                self.optimizer.zero_grad()
                # 修改这里，正确传递 x 和 edge_index
                output = self.model(batch.x, batch.edge_index)
                loss = self.loss_fn(output, batch.y)
                loss.backward()
                self.optimizer.step()
        return self.model.state_dict()
