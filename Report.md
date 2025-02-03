# Project Report

## Deep Deterministic Policy Gradient (DDPG) Overview

Q-Learning is a reinforcement learning algorithm where an agent learns to explore an environment by taking actions that change the state of the environment. For each action the agent takes, it receives a reward, which can be:

- **Negative**: when the action is bad;
- **Positive**: when the action is good;
- **Neutral (zero)**: when the action has no significant impact.

The chosen action is based on the highest Q-value, calculated using the **Bellman equation**:

```math
Q(s,a) \gets Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]
```

### Where:
- `s`: Current state;
- `a`: Action taken;
- `r`: Reward received;
- `s'`: New state reached after the action;
- `\alpha`: Learning rate;
- `\gamma`: Discount factor, weighting future rewards.

The Q-value represents the expected future reward for taking action `a` in state `s`. These values are stored in a table that maps states to possible actions. However, the main disadvantage of Q-Learning is its **lack of scalability**, as it requires knowledge of all possible states in the environment.

---

## Deep Q-Network (DQN)

DQN addresses the scalability issue of Q-Learning by replacing the Q-value table with a **neural network**. This network:

- Receives the state as **input**;
- Outputs the Q-values for each possible action.

In a **greedy policy**, the action with the highest Q-value is selected. This eliminates the need to know all possible states beforehand.

### Neural Network Functionality in DQN

A neural network in DQN operates as follows:
1. **Input layer**: Receives the state of the environment.
2. **Hidden layers**: Apply weights and activation functions to process the input.
3. **Output layer**: Produces the Q-values for the corresponding actions.

Learning is performed using the **backpropagation method** to minimize the error between predicted and target Q-values.

---

## Improvements to DQN

Despite its efficiency, the simple DQN model is prone to instability. To address this, several improvements have been introduced:

### 1. Target Neural Network
In the simple DQN, updating both the neural network weights and the Q-values simultaneously leads to instability. A **target network** is used to solve this problem:

- The target network has the same architecture as the main network but with weights "frozen" for a few episodes.
- The main network updates its weights based on Q-values provided by the target network.
- Periodically, the weights of the target network are synchronized with those of the main network, improving stability.

### 2. Experience Replay Buffer
To prevent the network from overfitting to sequential patterns in the data, transitions `(s, a, r, s')` are stored in an **experience buffer**:

- Transitions are stored in the buffer.
- During training, random samples from the buffer are used, promoting diversity in the training data and improving efficiency.

A code for the DQN with Replay Buffer is show below [(Source)](https://arxiv.org/abs/1312.5602).

```plaintext
Initialize replay memory D to capacity N
Initialize action-value function Q with random weights

For episode = 1, M do
    Initialize sequence s₁ = {x₁} and preprocessed sequence ϕ₁ = ϕ(s₁)

    For t = 1, T do
        With probability ε, select a random action aₜ
        Otherwise, select aₜ = argmaxₐ Q(ϕ(sₜ), a; θ)

        Execute action aₜ in emulator and observe reward rₜ and image xₜ₊₁
        Set sₜ₊₁ = sₜ, aₜ, xₜ₊₁ and preprocess ϕₜ₊₁ = ϕ(sₜ₊₁)
        Store transition (ϕₜ, aₜ, rₜ, ϕₜ₊₁) in D

        Sample random minibatch of transitions (ϕⱼ, aⱼ, rⱼ, ϕⱼ₊₁) from D
        Set yⱼ = 
            rⱼ                            if terminal ϕⱼ₊₁
            rⱼ + γ maxₐ' Q(ϕⱼ₊₁, a'; θ⁻)  for non-terminal ϕⱼ₊₁

        Perform a gradient descent step on 
        (yⱼ - Q(ϕⱼ, aⱼ; θ))² according to Equation 3
    End For
End For
```

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



