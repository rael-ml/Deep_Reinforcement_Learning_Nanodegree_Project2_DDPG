# DDPG Application with Unity ML-Agents for Udacity's Deep Reinforcement Learning Nanodegree

## Project Overview
In this project, the goal is to train an agent using Deep Q-Networks (DQN) to navigate a 3D environment and gather bananas. The environment, provided by Unity ML-Agents, features a grid where the agent can collect yellow bananas for a reward and blue bananas for a penalty. The agent’s objective is to maximize the collection of yellow bananas while avoiding blue ones.

A sample of the environment can be seen below:

![Banana Environment](https://video.udacity-data.com/topher/2018/June/5b1ab4b0_banana/banana.gif)

## Problem Description
The agent interacts with a virtual environment, where the state space consists of 37 dimensions, including the agent's velocity and sensor inputs from ray-based vision. The agent must make decisions based on these observations. The action space consists of four discrete actions:

- 0: Move forward
- 1: Move backward
- 2: Turn left
- 3: Turn right

The agent is rewarded with +1 for collecting yellow bananas and penalized with -1 for picking up blue bananas. The task is episodic, and the agent must achieve an average score of +13 over 100 consecutive episodes to successfully solve the environment.

## Files Included
- **Navigation.ipynb**: Jupyter Notebook containing the implementation of the DQN agent for the Banana environment.
- **README.md**: file with details about the project and setup instructions.
- **checkpoint.pth**: Saved weights of the trained DQN network that can solve the environment.
- **requirements.txt**: Python dependencies required to run the project.
- **setup.py**: Script for installing the Unity Machine Learning Agents library and related dependencies.
- **Report.md**: A detailed report on the learning algorithm, hyperparameters, and ideas for future improvements.
- **experimentplots**: Plot of the rewards of the fours experiments detailed in Report.md.

## Getting Started

To run the code, you'll need to download the Banana environment from one of the following links based on your operating system:

- **Linux**: [Download Link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- **Mac OSX**: [Download Link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
- **Windows (32-bit)**: [Download Link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
- **Windows (64-bit)**: [Download Link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

Once downloaded, unzip the file and place it in the project folder.

## Installation Instructions

1. First, install the necessary dependencies by running the following command:

<pre> python  !pip -q install . </pre>

2. This will trigger the setup.py script and install the required Python libraries listed in requirements.txt.

3. After installation, restart the kernel in your Jupyter Notebook environment.

4. Open Navigation.ipynb to begin running the DDPG agent on the Reacher environment.

