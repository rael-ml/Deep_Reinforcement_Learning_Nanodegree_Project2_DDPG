# Project Report

## Deep Deterministic Policy Gradient (DDPG) Overview

Reinforcement Learning Methods can be Value Based or Policy based. The first one aims to estimate the expected sum of rewards obtained by taking an action given a state. In a value-based deep reinforcement learning, the agent chooses the action with be best Q-value (expected sum of rewards). In Policy bases methods, the neural network predictd directly the action that must be taken given a state. Deep Deterministic Policy Gradient is a powerful reinforcement learning algorithm that has an Action Network (Policy Based) and an Critic Network (Value based), so in this way, it directly predicts that action through the Actor Network but that action is evalueated by the Critic.  



## Experiments
The table below summarizes the results of multiple training runs with different hyperparameters:

| Parameter               | Experiment 1     | Experiment 2     | Experiment 3    | Experiment 4    | Experiment 5        |
|-------------------------|-----------|-----------|-----------|-----------|---------------|
| BUFFER_SIZE            | 100,000   | 100,000   | 100,000   | 100,000   | 100,000       |
| BATCH_SIZE             | 64        | 64        | 128       | 64        | 64            |
| GAMMA                  | 0.99      | 0.9       | 0.9       | 0.9       | 0.9           |
| TAU                    | 1.00E-03  | 1.00E-03  | 1.00E-03  | 1.00E-03  | 1.00E-03      |
| LR (Learning Rate)     | 5.00E-04  | 5.00E-04  | 5.00E-04  | 5.00E-04  | 5.00E-03      |
| UPDATE_EVERY           | 4         | 4         | 4         | 4         | 16            |
| eps_end                | 0.01      | 0.01      | 0.01      | 0.01      | 0.01          |
| n_hidden_layers        | 2         | 2         | 2         | 2         | 2             |
| fc1_units              | 256       | 256       | 256       | 128       | 128           |
| fc2_units              | 128       | 128       | 128       | 128       | 128           |
| N° of episodes to solve  | 537       | 503       | 611       | 545       | Not Solved    |
| Average Score          | 13.04     | 13        | 13        | 13.02     | Not Solved    |


In the first three experiments, we used a neural network with an architecture of 37 (state_size) x 256 x 156 x 4 (action_size).

Experiment 1: The agent successfully solved the problem in just 537 episodes.

Experiment 2: We reduced the gamma parameter (which controls how much future rewards are valued). As a result, the agent solved the environment in 503 episodes, a slight improvement compared to the first experiment.

Experiment 3: The batch size (number of samples drawn from the replay buffer for training) was doubled. With more data to train on, the network took 611 episodes to solve the environment, demonstrating that a larger batch size can sometimes slow convergence

Experiment 4: The size of the first hidden layer was reduced to 128, which resulted in the environment being solved in 545 episodes. This showed that it is possible to reduce the network's complexity without significantly affecting its performance

Experiment 5: The learning rate was increased to 5e^-3, but the network failed to improve. It stagnated at an average reward of 10.39 and was unable to solve the environment.

This was a very simple analysis of the hyperparameters, providing only a preliminary understanding of their behavior. A more detailed investigation, involving extensive exploration of various hyperparameters, repeated experiments, collection of averages, and hypothesis testing, would be necessary to obtain clearer insights into the impact of hyperparameter tuning.

### Reward Plots During Training Across Four Experiments

Below are the reward plots observed during training for four different experimental setups. These plots illustrate the performance improvements across episodes.

<img src="rewardsplot.png" alt="Learning Plot" width="800">


## Ideas for Future Work
As suggestions for future work, in addition to a more advanced study of hyperparameters, we propose applying more recent methodologies in Deep Q-Networks (DQNs) to assess potential performance improvements. These could include prioritized replay buffer, double-DQN, or DQN Rainbow. Another possibility is to explore policy gradient methods, such as Proximal Policy Optimization (PPO).



