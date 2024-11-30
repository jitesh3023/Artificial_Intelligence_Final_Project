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
GAMMA = 0.99 # γ
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 5000  # Adjusted for smoother decay
target_update = 10
memory_capacity = 10000
num_episodes = 500  # At least 300 episodes to see loss decline
learning_rate = 1e-3


# Epsilon-greedy action selection
def select_action(env, policy_net:DQN, state, steps_done):
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)  # probability of choosing random action, decay over time
    if random.random() < eps_threshold: # sample random action
        return env.action_space.sample()
    else:
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = policy_net(state)
            return q_values.argmax().item()  # Index of max q-value


# Optimize the model
def optimize_model(memory:ReplayMemory, policy_net:DQN, target_net:DQN, optimizer:torch.optim):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    batch = list(zip(*transitions))

    #* somehow batch[0] isn't consistent. It should be np.ndarray, but sometime it has an extra empty dict and became tuple.
    #* This part is a temp workaround to fix datatype inconsistency.
    batch0 = list()
    for i in range(len(batch[0])):
        if not isinstance(batch[0][i], np.ndarray):
            batch0.append(batch[0][i][0].copy())
        else:
            batch0.append(batch[0][i].copy())

    # state_batch = torch.tensor(np.array(batch[0]), dtype=torch.float32) # [[x128]x64]   #! issue at np.array(batch[0])
    #! ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (64,) + inhomogeneous part.
    state_batch = torch.tensor(np.array(batch0), dtype=torch.float32)
    action_batch = torch.tensor(batch[1], dtype=torch.int64).unsqueeze(1)
    reward_batch = torch.tensor(batch[2], dtype=torch.float32).unsqueeze(1)
    next_state_batch = torch.tensor(np.array(batch[3]), dtype=torch.float32)
    done_batch = torch.tensor(batch[4], dtype=torch.float32).unsqueeze(1)

    # Compute Q(s_t, a)
    q_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states
    next_q_values = target_net(next_state_batch).max(1)[0].detach().unsqueeze(1)

    # Compute the expected Q values
    expected_q_values = reward_batch + (GAMMA * next_q_values * (1 - done_batch))

    # Compute Huber loss
    # criterion = nn.SmoothL1Loss()     # works bad
    criterion = nn.MSELoss()
    loss = criterion(q_values, expected_q_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1)  # Gradient clipping
    optimizer.step()

    return loss



def dqn_run():
    # Initialize policy and target networks
    # grocery_list = ['Cheese', 'Broccoli', 'Apple', 'Nuts', 'Chicken'] # for test
    env = GroceryStoreEnv()
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    policy_net = DQN(input_dim, output_dim)
    target_net = DQN(input_dim, output_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    memory = ReplayMemory(memory_capacity)


    # Training/Learning
    reward_per_episode = []
    loss_per_episode = []
    steps_done = 0
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_loss = 0
        total_reward = 0
        done = False

        step_limit = 0  # Limit the number of steps per episode
        while not done or step_limit > 500:
            action = select_action(env, policy_net, state, steps_done)
            steps_done += 1
            next_state, reward, done, truncated, emptydict = env.step(action)
            done = done or truncated
            total_reward += reward

            memory.push(state, action, reward, next_state, done)
            state = next_state

            loss = optimize_model(memory, policy_net, target_net, optimizer)
            if loss != None:
                total_loss += loss.item()

            step_limit += 1


        # Update the target network
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        print(f"Episode {episode}, Total reward: {total_reward}\n")

        # Record reward & loss
        reward_per_episode.append(total_reward)
        loss_per_episode.append(total_loss)


    # Plot the total rewards after training
    fig, ax = plt.subplots(nrows=2, ncols=1)
    fig.subplots_adjust(hspace=0.3)
    ax[0].plot(reward_per_episode)
    ax[0].set(xlabel='Episode', ylabel='Reward', title='Reward over Episode')
    ax[1].plot(loss_per_episode)
    ax[1].set(xlabel='Episode', ylabel='Loss', title='Loss over Episode')
    plt.show()


    # Visualize the trained agent
    state, _ = env.reset()
    env.render()
    for t in range(500):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = policy_net(state_tensor)
            action = q_values.argmax().item()
        next_state, reward, done, truncated, emptydict = env.step(action)
        env.render()
        state = next_state
        if done:
            break

    env.close()


dqn_run()