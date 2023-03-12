import torch
import torch.nn.functional as F

class Agent:
    def __init__(self, net):
        self.net = net

    def act(self, state):
        state = torch.FloatTensor(state)
        logits = self.net(state)
        probs = F.softmax(logits, dim=1)
        action = probs.multinomial(num_samples=1)
        return action.item()

    def evaluate(self, state, action):
        state = torch.FloatTensor(state)
        logits = self.net(state)
        log_probs = F.log_softmax(logits, dim=1)
        action_log_probs = log_probs.gather(1, torch.LongTensor(action).unsqueeze(1)))
        return action_log_probs.item()

    def update(self, loss):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
