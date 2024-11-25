import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(64, 1)

    def forward(self, state):
        x = self.base(state)
        return self.actor(x), self.critic(x)


class PPOAgent:
    def __init__(self, state_dim, action_dim, clip_epsilon=0.2, lr=3e-4):
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.clip_epsilon = clip_epsilon
        self.gamma = 0.99
        self.lam = 0.95
        self.loss_log = []  
        self.timesteps = 0 

    def compute_advantages(self, rewards, values, dones):
        advantages = []
        advantage = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            advantage = delta + self.gamma * self.lam * (1 - dones[t]) * advantage
            advantages.insert(0, advantage)
        return advantages

    def update(self, states, actions, log_probs, rewards, values, dones):
        advantages = self.compute_advantages(rewards, values, dones)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + torch.tensor(values[:-1], dtype=torch.float32)

        for _ in range(4):  # PPO epochs
            action_probs, new_values = self.model(torch.tensor(states, dtype=torch.float32))
            action_dist = torch.distributions.Categorical(action_probs)
            new_log_probs = action_dist.log_prob(torch.tensor(actions, dtype=torch.float32))
            entropy = action_dist.entropy()

            ratios = torch.exp(new_log_probs - torch.tensor(log_probs, dtype=torch.float32))

            # Clipped surrogate objective and policy loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = nn.MSELoss()(new_values.squeeze(), returns)

            # Total loss with entropy bonus
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Log the loss and timestep
            self.loss_log.append((self.timesteps, loss.item()))

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        action_probs, value = self.model(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action).item(), value.item()


def preprocess_state(state):
    """Flatten the state if it is a dictionary or structured observation."""
    if isinstance(state, dict):
        state = np.concatenate([np.ravel(v) for v in state.values()])
    elif isinstance(state, (list, tuple)): 
        state = np.concatenate([np.ravel(np.array(s)) for s in state])
    return np.array(state, dtype=np.float32)


def collect_trajectories(env, agent, steps=2048):
    states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
    state, _ = env.reset()  
    state = preprocess_state(state)
    
    for _ in range(steps):
        action, log_prob, value = agent.select_action(state)
        next_state, reward, done, truncated, _ = env.step(action) 
        done = done or truncated

        next_state = preprocess_state(next_state)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        log_probs.append(log_prob)
        values.append(value)
        dones.append(done)

        state = next_state
        if done:
            state, _ = env.reset()  # Reset the environment
            state = preprocess_state(state)

    values.append(agent.model(torch.tensor(state, dtype=torch.float32))[1].item())
    return states, actions, rewards, log_probs, values, dones


def train(env, agent, iterations=1000):
    for iteration in range(1, iterations + 1):
        # Collect trajectories
        states, actions, rewards, log_probs, values, dones = collect_trajectories(env, agent)
        
        # Update agent with the trajectories and log loss
        agent.update(states, actions, log_probs, rewards, values, dones)
        
        print(f"Iteration {iteration}: Policy updated.")
    
    # Extract the iteration numbers and loss values from loss_log
    iteration_numbers = list(range(1, len(agent.loss_log) + 1))
    loss_values = [loss for _, loss in agent.loss_log]

    # Plotting iterations vs. loss
    plt.figure(figsize=(10, 6))
    plt.plot(iteration_numbers, loss_values, label="Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training Loss vs. Iterations")
    plt.legend()
    plt.grid()
    plt.show()




def evaluate(env, agent, episodes=5):
    for episode in range(episodes):
        state, _ = env.reset()
        state = preprocess_state(state)
        total_reward = 0
        done = False

        while not done:
            action, _, _ = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated

            state = preprocess_state(next_state)
            total_reward += reward

            env.render()  

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")


if __name__ == "__main__":
    from environment import GroceryStoreEnv  

    env = GroceryStoreEnv(grocery_list=["Butter", "Potatoes", "Crackers"])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPOAgent(state_dim, action_dim)

    # Training or evaluating
    choice = input("Train or evaluate? (train/eval): ").strip().lower()
    if choice == "train":
        train(env, agent, iterations=1000)
        torch.save(agent.model.state_dict(), "ppo_agent_scratch.pth")
    elif choice == "eval":
        agent.model.load_state_dict(torch.load("ppo_agent_scratch.pth"))
        evaluate(env, agent, episodes=5)
