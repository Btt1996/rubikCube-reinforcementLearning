# rubikCube-reinforcementLearning
 Rubik's Cube Solver using Reinforcement Learning

This repository contains code for training and evaluating a Rubik's Cube solver using reinforcement learning. 

## Files

* `agent.py` - defines the reinforcement learning agent
* `environment.py` - defines the Rubik's Cube environment
* `main.py` - main script to train and evaluate the Rubik's Cube solver
* `neural.py` - defines the neural network architecture for the agent
* `optimiser.py` - defines the optimizer used to train the agent
* `policy.py` - defines the policy used by the agent to select actions
* `trainer.py` - defines the training process for the agent
* `evaluator.py` - defines the evaluation process for the agent
* `trainer_options.py` - defines options and hyperparameters for the training process
* `evaluator_options.py` - defines options and hyperparameters for the evaluation process
* `utils.py` - utility functions used throughout the project

## Database

A database can be useful for storing and retrieving information about the training process and the agent's performance. For example, you could use a database to store:

* The agent's current state and weights
* The agent's performance metrics during training (e.g., average reward, success rate, etc.)
* The configuration settings for the agent and environment

This information can then be used to resume training from a previous checkpoint or to compare the performance of different agents or configurations.

One popular database for machine learning projects is SQLite, which is a lightweight, file-based database that can be easily integrated into Python projects. Other options include MySQL, PostgreSQL, and MongoDB.

If you decide to add a database to this repository, be sure to include any necessary installation and setup instructions in the README.md file.

## Contributing

Contributions to this project are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue on GitHub.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.
