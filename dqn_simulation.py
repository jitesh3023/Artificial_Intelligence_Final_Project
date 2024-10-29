import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from environment import GroceryStoreEnv
import matplotlib.pyplot as plt

# Set matplotlib backend if necessary (uncomment if you encounter backend issues)
import matplotlib
matplotlib.use('TkAgg')

# Define the neural network for the DQN agent
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # First fully connected layer with 128 units
        self.fc2 = nn.Linear(128, 128)        # Second fully connected layer with 128 units
        self.fc3 = nn.Linear(128, output_dim) # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply ReLU activation function
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Experience Replay Memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(tuple(args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Hyperparameters
batch_size = 64
gamma = 0.99
eps_start = 1.0
eps_end = 0.01
eps_decay = 5000  # Adjusted for smoother decay
target_update = 10
memory_capacity = 10000
num_episodes = 500  # You can reduce this number to see results sooner
learning_rate = 1e-3

# Initialize policy and target networks
env = GroceryStoreEnv()
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

policy_net = DQN(input_dim, output_dim)
target_net = DQN(input_dim, output_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
memory = ReplayMemory(memory_capacity)

# Epsilon-greedy action selection
def select_action(state, steps_done):
    eps_threshold = eps_end + (eps_start - eps_end) * np.exp(-1. * steps_done / eps_decay)
    if random.random() < eps_threshold:
        return env.action_space.sample()  # Explore
    else:
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = policy_net(state)
            return q_values.argmax().item()  # Exploit

# Optimize the model
def optimize_model():
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    batch = list(zip(*transitions))

    state_batch = torch.tensor(np.array(batch[0]), dtype=torch.float32)
    action_batch = torch.tensor(batch[1], dtype=torch.int64).unsqueeze(1)
    reward_batch = torch.tensor(batch[2], dtype=torch.float32).unsqueeze(1)
    next_state_batch = torch.tensor(np.array(batch[3]), dtype=torch.float32)
    done_batch = torch.tensor(batch[4], dtype=torch.float32).unsqueeze(1)

    # Compute Q(s_t, a)
    q_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states
    next_q_values = target_net(next_state_batch).max(1)[0].detach().unsqueeze(1)

    # Compute the expected Q values
    expected_q_values = reward_batch + (gamma * next_q_values * (1 - done_batch))

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(q_values, expected_q_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1)  # Gradient clipping
    optimizer.step()

# Initialize total_rewards list for plotting
total_rewards = []

# Training loop
steps_done = 0
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    for t in range(500):  # Limit the number of steps per episode
        action = select_action(state, steps_done)
        steps_done += 1
        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        memory.push(state, action, reward, next_state, done)
        state = next_state

        optimize_model()

        if done:
            break
    # Update the target network
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())
    print(f"Episode {episode}, Total reward: {total_reward}")

    # Append total_reward to the list
    total_rewards.append(total_reward)

# Plot the total rewards after training
plt.figure()
plt.plot(total_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Progress')
plt.show()

# Visualize the trained agent
state = env.reset()
env.render()
for t in range(500):
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = policy_net(state_tensor)
        action = q_values.argmax().item()
    next_state, reward, done, _ = env.step(action)
    env.render()
    state = next_state
    if done:
        break

env.close()
