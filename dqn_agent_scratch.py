import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from environment import GroceryStoreEnv  # Import your existing environment
import matplotlib.pyplot as plt

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state):
        return self.model(state)


class DQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-3, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQN(state_dim, action_dim).to(self.device)
        self.target_model = DQN(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # Initialize replay buffer
        self.replay_buffer = deque(maxlen=100000)
        self.batch_size = 64

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def select_action(self, state, evaluate=False):
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)  # Explore
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()  # Exploit

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def sample_batch(self):
        return random.sample(self.replay_buffer, self.batch_size)

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.sample_batch()
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Compute current Q values
        q_values = self.model(states).gather(1, actions)

        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1, keepdim=True)[0]
            q_targets = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss
        loss = self.criterion(q_values, q_targets)

        # Update model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        return loss


def preprocess_state(state):
    """Flatten the state if it is a dictionary or structured observation."""
    if isinstance(state, dict):
        state = np.concatenate([np.ravel(v) for v in state.values()])
    elif isinstance(state, (list, tuple)):  # Handle lists/tuples
        state = np.concatenate([np.ravel(np.array(s)) for s in state])
    return np.array(state, dtype=np.float32)


def train_dqn(env, agent, episodes=500, update_target_every=10, visualize=True):
    rewards_per_episode = []
    losses_per_episode = []  # Track loss per episode

    for episode in range(episodes):
        state, _ = env.reset()
        state = preprocess_state(state)
        total_reward = 0
        total_loss = 0  # Track total loss for the episode
        done = False

        while not done:
            if visualize:  # Render the environment during training
                env.render()

            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            next_state = preprocess_state(next_state)

            # Store transition and train
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.train()  # Get the loss from training (modify the `train` method to return loss)
            if loss is not None:
                total_loss += loss.item()  # Accumulate loss for this episode

            state = next_state
            total_reward += reward

        rewards_per_episode.append(total_reward)
        losses_per_episode.append(total_loss)

        if episode % update_target_every == 0:
            agent.update_target_network()

        print(f"Episode {episode + 1}: Total Reward = {total_reward}, Loss = {total_loss:.3f}, Epsilon = {agent.epsilon:.3f}")

    # Plot the training loss vs episodes
    plt.figure(figsize=(10, 6))
    plt.plot(range(episodes), losses_per_episode, label="Loss")
    plt.xlabel("Episodes")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Episodes")
    plt.legend()
    plt.grid()
    plt.show()

    return rewards_per_episode, losses_per_episode



def evaluate_dqn(env, agent, episodes=5, visualize=True):
    for episode in range(episodes):
        state, _ = env.reset()
        state = preprocess_state(state)
        total_reward = 0
        done = False

        while not done:
            if visualize:  # Render the environment during evaluation
                env.render()

            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            next_state = preprocess_state(next_state)

            state = next_state
            total_reward += reward

        print(f"Evaluation Episode {episode + 1}: Total Reward = {total_reward}")


if __name__ == "__main__":
    env = GroceryStoreEnv(grocery_list=["Butter", "Potatoes", "Crackers"])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)

    # Ask the user whether to train or evaluate the agent
    choice = input("Train or evaluate? (train/eval): ").strip().lower()
    if choice == "train":
        train_dqn(env, agent, episodes=100, visualize=False)  # Disable visualization for training
        torch.save(agent.model.state_dict(), "dqn_agent.pth")
        print("Agent trained and saved as 'dqn_agent.pth'.")
    elif choice == "eval":
        agent.model.load_state_dict(torch.load("dqn_agent.pth"))
        print("Agent loaded. Evaluating...")
        evaluate_dqn(env, agent, episodes=5, visualize=True)  # Enable visualization for evaluation