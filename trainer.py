import torch.optim as optim
from utils import ReplayBuffer

class Trainer:
    def __init__(self, net, optimizer_class, criterion, device):
        self.device = device
        self.net = net.to(device)
        self.optimizer = optimizer_class(net.parameters())
        self.criterion = criterion
        self.replay_buffer = ReplayBuffer()
    
    def train_step(self, batch_size, gamma):
        if len(self.replay_buffer) < batch_size:
            return
        transitions = self.replay_buffer.sample(batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.FloatTensor(batch.state).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(batch.next_state).to(self.device)
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state))).to(self.device)
        non_final_next_states = torch.FloatTensor([s for s in batch.next_state if s is not None]).to(self.device)

        state_action_values = self.net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(batch_size).to(self.device)
        next_state_values[non_final_mask] = self.net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * gamma) + reward_batch

        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.agent.update(loss)
