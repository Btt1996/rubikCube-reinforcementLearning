# optimizer.py

import torch.optim as optim

class AdamOptimizer:
    def __init__(self, params, lr=0.001, weight_decay=0.0):
        self.optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()
