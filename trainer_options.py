# trainer_options.py

class TrainerOptions:
    def __init__(self, num_episodes=10000, max_steps=50, gamma=0.99, batch_size=32, target_update_freq=100, 
                 memory_capacity=10000, initial_memory_size=1000, learning_rate=0.001, weight_decay=0.0):
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.memory_capacity = memory_capacity
        self.initial_memory_size = initial_memory_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
