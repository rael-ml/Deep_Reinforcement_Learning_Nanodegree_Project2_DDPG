# Project Report

## Deep Deterministic Policy Gradient (DDPG) Overview

Reinforcement Learning Methods can be Value Based or Policy based. The first one aims to estimate the expected sum of rewards obtained by taking an action given a state. In a value-based deep reinforcement learning, the agent chooses the action with be best Q-value (expected sum of rewards). In Policy bases methods, the neural network predictd directly the action that must be taken given a state. Deep Deterministic Policy Gradient is a powerful reinforcement learning algorithm that has an Action Network (Policy Based) and an Critic Network (Value based), so in this way, it directly predicts the action through the Actor Network but that action is evalueated by the Critic.  

DDPG is a off-policy algorithm, which means that the policy learned is not tha same as the policy in wich it was trained. DDPG is suitable for a continuos action space.

In this algorithm, in each step, the agent save the (state, action, reward, new state) tuple in the experience buffer and if the experience buffer is greater than the batch size, the agent learns from that experience. The learing process of de DDPG consists of a Actor Neural Network and a Critic Network. The Actor has the state as input and the action as the output. The Critic also has the state as input but has the expectated reward as output, as a way to measure the quality of the action taken. To stabilize training, both actor and critcs have a target network, with the same architecture and in each new step the target networks are soft updated. 

The Critic error is measuered by the difference between Q-value outputted by the local target and the Belmann Equation utilizing the Q-value of the target network. This makes sense because it is expected that the local network learns to estimate de expected return of the future actions (which is calculated by the bellmann equation). The Actor error is measured by passing the state predicted by the local actor to the local critic (in order to evaluated if it was a good action). 

**Bellman equation** to update the critic network:

```math
Q_{target} = r + (\gamma * Q_{targets\_next} * (1 - dones))
```


## Improvements to the Neural Networks

Despite its efficiency, the simple neural network model is prone to instability. To address this, several improvements have been introduced in the DDPG:

### 1. Target Neural Network
In the simple network, updating both the neural network weights and the output values simultaneously leads to instability. A **target network** is used to solve this problem:

- The target network has the same architecture as the main network but with weights "frozen" for a few episodes.
- The main network updates its weights based on the outputs provided by the target network.
- Periodically, the weights of the target network are synchronized with those of the main network, improving stability.
- The Actor has a local and a target network and the Critic also has a local and a taget network.

### 2. Experience Replay Buffer
To prevent the network from overfitting to sequential patterns in the data, transitions `(s, a, r, s')` are stored in an **experience buffer**:

- Transitions are stored in the buffer.
- During training, random samples from the buffer are used, promoting diversity in the training data and improving efficiency.

### 3. Ornstein–Uhlenbeck process
The Ornsteiun-Uhlenbeck noise is added to the output of the network to act as a exploration term. This kind of noise is suitable to continuos action spaces because it is time-correlated, so it is expected that the model will not suggest a action with a great difference than the acton before.


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

<img src="rewards.png" alt="Learning Print" width="400">

<img src="rewardsplot.png" alt="Learning Plot" width="800">


## Ideas for Future Work
For future improvements, beyond a more in-depth study of hyperparameters, we suggest exploring more stable methodologies such as:  
- **Trust Region Policy Optimization (TRPO)**  
- **Truncated Natural Policy Gradient (TNPG)**  
- **Distributed Distributional Deep Deterministic Policy Gradient (D4PG)** (a state-of-the-art approach)  

Additionally, an interesting direction for future work would be comparing the results obtained in this single-agent scenario with the multi-agent Reacher environment to evaluate performance differences.  



