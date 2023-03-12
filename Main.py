import torch
from environment import RubikCubeEnvironment
from agent import RubikCubeAgent
from trainer import RubikCubeTrainer
from evaluator import RubikCubeEvaluator

# Set hyperparameters
learning_rate = 0.001
num_epochs = 100
batch_size = 64

# Create environment
env = RubikCubeEnvironment()

# Create agent
agent = RubikCubeAgent()

# Create trainer
trainer = RubikCubeTrainer(agent=agent, learning_rate=learning_rate, batch_size=batch_size)

# Train agent
trainer.train(env=env, num_epochs=num_epochs)

# Create evaluator
evaluator = RubikCubeEvaluator(agent=agent)

# Evaluate agent
success_rate = evaluator.evaluate(env=env, num_trials=1000)
print("Success rate: ", success_rate)
