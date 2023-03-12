# policy.py

import numpy as np

class EpsilonGreedyPolicy:
    def __init__(self, num_actions, initial_epsilon=1.0, final_epsilon=0.01, decay_rate=0.999):
        self.num_actions = num_actions
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.decay_rate = decay_rate
        self.epsilon = initial_epsilon

    def select_action(self, q_values):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(q_values)

    def update_epsilon(self, episode):
        self.epsilon = max(self.final_epsilon, self.initial_epsilon * self.decay_rate**episode)
