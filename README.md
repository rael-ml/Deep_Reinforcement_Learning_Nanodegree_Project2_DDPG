# DDPG Application with Unity ML-Agents for Udacity's Deep Reinforcement Learning Nanodegree

## Project Overview
In this project, the goal is to train an agent using **Deep Deterministic Policy Gradient (DDPG)** to control a virtual robotic arm and make it reach a moving target.  

Unity ML-Agents provides two versions of this environment:  
- **Single-agent environment** – where only one robotic arm operates.  
- **Multi-agent environment** – where multiple robotic arms operate in parallel.  

For simplicity, this project focuses on the **single-agent environment**. Below is a sample of the **multi-agent environment**. The single-agent version is identical, except that it contains only one robotic arm and target. 

![Reacher Environment](https://video.udacity-data.com/topher/2018/June/5b1ea778_reacher/reacher.gif)

## Problem Description
The agent interacts with a virtual environment where the **state space** consists of **33 dimensions**, including:  
- Position  
- Rotation  
- Velocity  
- Angular velocity of the arm  

Based on these observations, the agent must decide how to move its joints toward the target.  

The **action space** consists of **four continuous actions**, representing the torque applied to two joints, with values ranging from **-1 to 1**.  

- The agent receives a **reward of +0.1** for each time step its hand remains at the goal location.  
- The objective is to keep the hand at the target position for as many time steps as possible.   
- The task is **episodic**, meaning each episode has a defined start and end.  
- The environment is considered **solved** when the agent achieves an **average score of +30 over 100 consecutive episodes**.  

## Files Included
- **Continuos_Control.ipynb**: Jupyter Notebook containing the implementation of the DDPG agent for the Reacher environment.
- **README.md**: file with details about the project and setup instructions.
- **checkpoint_actor.pth**: Saved weights of the trained Actor network that can solve the environment.
- **checkpoint_critic.pth**: Saved weights of the trained Critic network that can solve the environment.
- **requirements.txt**: Python dependencies required to run the project.
- **setup.py**: Script for installing the Unity Machine Learning Agents library and related dependencies.
- **Report.md**: A detailed report on the learning algorithm, hyperparameters, and ideas for future improvements.
- **rewardsplot.png**: Plot of the rewards of the experiment detailed in Report.md.
- **rewards.png**: Print of the rewards of the experiment detailed in Report.md.

## Getting Started

To run the code, you'll need to download the reacher environment from one of the following links based on your operating system:

- **Linux**: [Download Link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
- **Mac OSX**: [Download Link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
- **Windows (32-bit)**: [Download Link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
- **Windows (64-bit)**: [Download Link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

Once downloaded, unzip the file and place it in the project folder.

## Installation Instructions

1. First, install the necessary dependencies by running the following command:

<pre> python  !pip -q install . </pre>

2. This will trigger the setup.py script and install the required Python libraries listed in requirements.txt.

3. After installation, restart the kernel in your Jupyter Notebook environment.

4. Open Continuos_Control.ipynb to begin running the DDPG agent on the Reacher environment.

