import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gym import spaces
from collections import deque
import matplotlib.pyplot as plt

# Import your environment
from environment import GroceryStoreEnv  # Ensure this points to your environment file

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## **Policy Network**
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        #print(f"Input to PolicyNetwork: {x.shape}")  # Debugging
        x = self.relu(self.fc1(x))
        #print(f"After fc1: {x.shape}")  # Debugging
        x = self.relu(self.fc2(x))
        #print(f"After fc2: {x.shape}")  # Debugging
        action_probs = self.softmax(self.fc3(x))
        #print(f"Action Probs: {action_probs.shape}")  # Debugging
        return action_probs

## **Value Network**
class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        state_values = self.fc3(x)
        return state_values

## **PPO Agent**
class PPOAgent:
    def __init__(self, env, gamma=0.99, lr=3e-4, epsilon=0.2, K_epochs=4, batch_size=64):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.K_epochs = K_epochs
        self.batch_size = batch_size

        # Set observation dimension to 8 (2 for robot position + 3 items x 2)
        self.obs_dim = 8
        self.act_dim = env.action_space.n

        # Initialize networks with correct input dimensions
        self.policy = PolicyNetwork(self.obs_dim, self.act_dim).to(device)
        self.value_net = ValueNetwork(self.obs_dim).to(device)

        # Optimizer setup
        self.optimizer = optim.Adam([
            {'params': self.policy.parameters(), 'lr': lr},
            {'params': self.value_net.parameters(), 'lr': lr}
        ])

        # Initialize old policy network for PPO updates
        self.policy_old = PolicyNetwork(self.obs_dim, self.act_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Loss function for value estimation
        self.MseLoss = nn.MSELoss()

    def process_observation(self, observation):
        # Extract robot position (first 2 elements)
        robot_position = observation[:2]
        
        # Extract all items' positions
        items_positions = observation[2:]
        items_list_keys = list(self.env.items_list.keys())
        
        # Extract positions of items in the grocery list
        selected_items_positions = []
        
        for item in self.env.grocery_list:
            if item in self.env.items_list:
                idx = items_list_keys.index(item)
                selected_items_positions.extend(items_positions[idx*2:(idx+1)*2])
            else:
                # Append zeros if item is not found
                selected_items_positions.extend([0, 0])
        
        # Ensure exactly 6 elements for items
        while len(selected_items_positions) < 6:
            selected_items_positions.extend([0, 0])
        
        # Combine robot position with selected items' positions
        processed_observation = np.array(robot_position.tolist() + selected_items_positions[:6], dtype=np.float32)
        
        return processed_observation

    def select_action(self, state):
       """
       Selects an action based on the current state using the policy network.
       
       Parameters:
           state (np.ndarray): The current state observation.
       
       Returns:
           tuple: (action, log probability of the action, entropy of the action distribution)
       """
       
       processed_state = self.process_observation(state)
       state_tensor = torch.FloatTensor(processed_state).to(device)
       state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension

       #print(f"State shape in select_action: {state_tensor.shape}")  # Debugging

       with torch.no_grad():
           action_probs = self.policy_old(state_tensor)
           dist = torch.distributions.Categorical(action_probs)
           action = dist.sample()
       return action.item(), dist.log_prob(action).item(), dist.entropy().item()

    def compute_returns(self, rewards, dones, next_value):
        """
        Computes the discounted returns.
        
        Parameters:
            rewards (list): List of rewards received.
            dones (list): List of done flags.
            next_value (float): The value of the next state.
        
        Returns:
            list: Discounted returns.
        """
        
        returns = []
        R = next_value

        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * (1 - dones[step])
            returns.insert(0, R)

        return returns

    def update(self, memory):
        """
        Updates the policy and value networks using the collected memory.
        
        Parameters:
            memory (dict): A dictionary containing states, actions, log probabilities, rewards, and done flags.
        """
        
        # Convert lists to tensors efficiently
        states = torch.FloatTensor(np.array(memory['states'])).to(device)
        actions = torch.LongTensor(memory['actions']).to(device)
        old_logprobs = torch.FloatTensor(memory['logprobs']).to(device)
        returns = torch.FloatTensor(memory['returns']).to(device)

        advantages = returns - self.value_net(states).detach().squeeze()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            action_probs = self.policy(states)
            dist = torch.distributions.Categorical(action_probs)
            logprobs = dist.log_prob(actions)
            entropy = dist.entropy()
            state_values = self.value_net(states).squeeze()

            # Finding the ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(logprobs - old_logprobs)

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages

            # Policy loss
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, returns) - 0.01 * entropy

            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Update the old policy with new policy weights after optimization step is complete.
        self.policy_old.load_state_dict(self.policy.state_dict())

    def train(self,max_episodes=100,max_steps=1000):
        """
        Trains the PPO agent.
        
        Parameters:
            max_episodes (int): Number of episodes to train.
            max_steps (int): Maximum number of steps per episode.
        """
        
        episode_rewards=[]
        memory={'states':[],'actions':[],'logprobs':[],'rewards':[],'dones':[]}

        for episode in range(1,max_episodes+1):

            print(f"Episode {episode}")
            state=self.env.reset()
            total_reward=0

            for step in range(max_steps):

                action ,logprob ,entropy=self.select_action(state)

                next_state ,reward ,done ,_=self.env.step(action)

                processed_state=self.process_observation(state)

                memory['states'].append(processed_state)

                memory['actions'].append(action)

                memory['logprobs'].append(logprob)

                memory['rewards'].append(reward)

                memory['dones'].append(done)

                state=next_state

                total_reward+=reward

                if done:

                    break

            with torch.no_grad():

                next_state_processed=self.process_observation(state)

                next_state_tensor=torch.FloatTensor(next_state_processed).to(device)

                next_state_tensor=next_state_tensor.unsqueeze(0) 

                next_value=self.value_net(next_state_tensor).item()

                returns=self.compute_returns(memory['rewards'],memory['dones'],next_value)

                memory['returns']=returns

            episode_rewards.append(total_reward)
            print(total_reward)
            memory={'states':[],'actions':[],'logprobs':[],'rewards':[],'dones':[]}

            if episode%10==0:

                avg_reward=np.mean(episode_rewards[-10:])

                #print(f"Episode {episode}\tAverage Reward: {avg_reward:.2f}")

        plt.plot(episode_rewards)

        plt.xlabel('Episode')

        plt.ylabel('Total Reward')

        plt.title('PPO Training Rewards')

        plt.show()


if __name__ == "__main__":
    env = GroceryStoreEnv()
    state = env.reset()
    env.grocery_list = ["Yogurt", "Bacon", "Potato Chips"]

    agent = PPOAgent(env)
    agent.train(max_episodes=1000)

    env.grocery_list = ["Yogurt", "Bacon", "Potato Chips"]

    done = False
    total_reward = 0

    while not done and not env.stop_simulation:
        action, _, _ = agent.select_action(state)
        state, reward, done, _ = env.step(action)
        # Render environment

        env.render()
        print(f"Action: {action}, Reward: {reward}, Observation: {state}, Done: {done}")

        #plt.pause(0.1)  # Non-blocking update

        total_reward += reward
        if done:
            print("Episode finished! Returning to entry/exit point.")
            env.reset()
            break 

    print(f"Total Reward after Training: {total_reward}")
    env.close()

print(f"Total Reward after Training: {total_reward}")


env.close()