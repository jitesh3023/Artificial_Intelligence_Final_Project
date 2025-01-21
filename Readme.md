# Grocery Store Reinforcement Learning Project

This project implements two reinforcement learning algorithms, **Proximal Policy Optimization (PPO)** and **Deep Q-Network (DQN)**, to navigate a simulated grocery store environment. Further for evaluating the performance we are comparing it with A star which acts as our ground truth.

---

## Installation

To run this project, ensure you have **Python 3.7+** installed. Use the following command to install the required libraries:

```bash
pip install numpy torch gymnasium matplotlib
```

## Usage

### Clone the Repository:

```bash
git clone https://github.com/user_name/Artificial_Intelligence_Final_Project.git)
cd Codes
```

### Run the PPO Agent::

```bash
python3 ppo_agent_scratch.py
```

### Run the DQN Agent::

```bash
python3 dqn_agent_scratch.py
```

### Run the A star Agent::

```bash
python3 a_star.py
```
If you plan to visualize the environment then we can do that just by running environment.py code. This will just open the environment and the agent will just roam around randomly.
### Run the A star Agent::

```bash
python3 environment.py
```

## Environment

The GroceryStoreEnv simulates a 20x20 grid world representing a grocery store. Key features include:

1. Various grocery items with specific locations
2. Shelf represented using black boxes
3. Robot represented using green colored block

# Algorithms

## PPO (Proximal Policy Optimization)

The PPO implementation includes:

- ActorCritic neural network with separate actor and critic heads
- GAE (Generalized Advantage Estimation) for computing advantages
- Clipped surrogate objective for policy updates

## DQN (Deep Q-Network)

The DQN implementation features:

- A neural network for Q-value approximation
- Experience replay buffer for off-policy learning
- Epsilon-greedy exploration strategy

## Pre-trained Weights

The project includes pre-trained weights for both PPO and DQN agents. These can be loaded for immediate evaluation without training.

## Running the Code

When running either ppo_agent.py or dqn_agent.py, you will be prompted to choose between training mode and evaluation mode:
Training mode:

- Trains a new agent from the start.
- Evaluation mode: Loads pre-trained weights and evaluates the agent's performance on our custom environment.

## License

```
MIT License

Copyright (c) 2024 Jitesh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
