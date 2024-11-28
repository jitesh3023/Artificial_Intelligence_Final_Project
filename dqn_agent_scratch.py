import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from environment import GroceryStoreEnv  
import matplotlib.pyplot as plt

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        # Creating a 2 layered neural network 

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
        self.gamma = gamma # discount factor
        self.epsilon = epsilon # because I am using epsilon greedy policy method
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQN(state_dim, action_dim).to(self.device)
        self.target_model = DQN(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # Initialize replay buffer
        # This is used for storing past experiences
        self.replay_buffer = deque(maxlen=100000)
        self.batch_size = 64

    def update_target_network(self):
        # Copying the weights from the main netwrok to the target network
        self.target_model.load_state_dict(self.model.state_dict())

    def select_action(self, state, evaluate=False):
        # Normal epsilon greedy policy
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)  # Explore
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()  # Exploit

    def store_transition(self, state, action, reward, next_state, done):
        # Storing the data in replay buffer
        self.replay_buffer.append((state, action, reward, next_state, done))

    def sample_batch(self):
        return random.sample(self.replay_buffer, self.batch_size)

    def train(self):
        # Train the DQN by sampling a batch and performing a Q-learning update.
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
    elif isinstance(state, (list, tuple)):  
        state = np.concatenate([np.ravel(np.array(s)) for s in state])
    return np.array(state, dtype=np.float32)


def train_dqn(env, agent, episodes=500, update_target_every=10, visualize=True):
    rewards_per_episode = []
    losses_per_episode = []  

    for episode in range(episodes):
        state, _ = env.reset()
        state = preprocess_state(state)
        total_reward = 0
        total_loss = 0
        done = False

        while not done:
            if visualize:  
                env.render()

            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            next_state = preprocess_state(next_state)

            # Store transition and train
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.train()  
            if loss is not None:
                total_loss += loss.item()  

            state = next_state
            total_reward += reward

        rewards_per_episode.append(total_reward)
        losses_per_episode.append(total_loss)

        if episode % update_target_every == 0:
            agent.update_target_network()

        print(f"Episode {episode + 1}: Total Reward = {total_reward}, Loss = {total_loss:.3f}, Epsilon = {agent.epsilon:.3f}")

    # Plotting the training loss vs episodes
    plt.figure(figsize=(10, 6))
    plt.plot(range(episodes), losses_per_episode, label="Loss")
    plt.xlabel("Episodes")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Episodes")
    plt.legend()
    plt.grid()
    plt.show()

    return rewards_per_episode, losses_per_episode



def evaluate_dqn(env, agent, episodes=5, visualize=True, plot_episode=1):
    cumulative_rewards = []  
    total_reward = 0

    for episode in range(episodes):
        state, _ = env.reset()
        state = preprocess_state(state)
        total_reward = 0
        step_rewards = []  
        done = False

        while not done:
            if visualize and episode + 1 == plot_episode: 
                env.render()

            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            next_state = preprocess_state(next_state)

            total_reward += reward
            step_rewards.append(total_reward) 

            state = next_state

        print(f"Evaluation Episode {episode + 1}: Total Reward = {total_reward}")

        if episode + 1 == plot_episode:
            cumulative_rewards = step_rewards

    # Plotting cumulative rewards for the plotted episode
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_rewards) + 1), cumulative_rewards, marker="o", label=f"Cumulative Reward (Episode {plot_episode})")
    plt.xlabel("Steps")
    plt.ylabel("Reward")
    plt.title(f" DQN -  Reward vs Steps")
    plt.legend()
    plt.grid()
    plt.show()



if __name__ == "__main__":
    # env = GroceryStoreEnv(grocery_list=["milk", "Eggs", "Cheese", "Yogurt", "Cream", "Butter", "Ice Cream",
    # "Potatoes", "Onions", "Tomatoes", "Lettuce", "Carrot", "Pepper", "Cucumbers", "Celery", "Broccoli", "Mushrooms", "Spinach", "Corn", "Cauliflower", "Garlic",
    # "Banana", "Berries", "Apple", "Grapes", "Melons", "Avocados", "Mandarins", "Oranges", "Peaches", "Pineapple", "Cherries", "Lemons", "Kiwis", "Mangoes",
    # "Baked Beans", "Black Beans", "Cookies", "Crackers", "Dried Fruits", "Gelatin", "Granola Bars", "Nuts", "Popcorn", "Potato Chips", "Pudding", "Raisins", "Pasta", "Peanut Butter",
    # "Chicken", "Lamb", "Bacon", "Ham", "Turkey", "Pork", "Sausage", 
    # "Aluminum Foil", "Garbage Bags", "Napkins", "Paper Plates", "Plastics Bags", "Straws", "Dish Soap"
    # ])
    env = GroceryStoreEnv(grocery_list=["Avocados", "Melons"])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)

    choice = input("Train or evaluate? (train/eval): ").strip().lower()
    if choice == "train":
        train_dqn(env, agent, episodes=1000, visualize=False)  
        torch.save(agent.model.state_dict(), "dqn_agent.pth")
        print("Agent trained and saved as 'dqn_agent.pth'.")
    elif choice == "eval":
        agent.model.load_state_dict(torch.load("dqn_agent.pth"))
        print("Agent loaded. Evaluating...")
        evaluate_dqn(env, agent, episodes=5, visualize=True)  