import matplotlib.pyplot as plt
import numpy as np
from environment import GroceryStoreEnv  # Import your GroceryStoreEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class LossLoggingCallback(BaseCallback):
    """Custom callback to log training loss over time."""
    def __init__(self):
        super(LossLoggingCallback, self).__init__()
        self.losses = []
        self.timesteps = []

    def _on_step(self) -> bool:
        # Log the current training loss
        if "train/loss" in self.model.logger.name_to_value:
            self.losses.append(self.model.logger.name_to_value["train/loss"])
            self.timesteps.append(self.num_timesteps)
        return True


def train_agent(env):
    """Train the PPO agent with loss logging."""
    print("Training the agent...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        clip_range=0.3,  # Increase exploration range
        ent_coef=0.05,  # Encourage exploration by increasing entropy
    )

    # Create and attach the loss logging callback
    loss_callback = LossLoggingCallback()
    model.learn(total_timesteps=500000, callback=loss_callback)  
    model.save("ppo_agent")  # Save the trained model
    print("Model trained and saved as 'ppo_agent'.")
    return model, loss_callback


def evaluate_agent(model, env, episodes=5):
    """Evaluate the trained agent with real-time rendering."""
    for ep in range(episodes):
        obs, _ = env.reset()  # Reset the environment for each episode
        terminated, truncated = False, False
        total_reward = 0

        print(f"\nStarting Episode {ep + 1}")

        # Render the initial state
        env.render()

        while not (terminated or truncated):
            # Predict and execute action
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            # Render the updated environment
            env.render()

            # Print debug information
            print(f"Position: {env.robot_position}, Reward: {reward}, Remaining Items: {env.grocery_list}")

        print(f"Episode {ep + 1} Total Reward: {total_reward}\n")
    
    env.close()  # Ensure environment is closed after evaluation



def plot_losses(loss_callback):
    """Plot the training loss over timesteps."""
    plt.figure(figsize=(10, 6))
    plt.plot(loss_callback.timesteps, loss_callback.losses, label="Training Loss")
    plt.xlabel("Timesteps")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Timesteps")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # Initialize the grocery store environment
    env = GroceryStoreEnv(grocery_list=["Butter", "Potatoes", "Crackers"])

    # Ask user whether to train the agent or load an existing model
    choice = input("Do you want to train the agent? (y/n): ").strip().lower()
    if choice == "y":
        # Train the agent and log losses
        model, loss_callback = train_agent(env)

        # Plot the training loss
        plot_losses(loss_callback)
    else:
        # Load the pretrained model
        model = PPO.load("ppo_agent")

    # Evaluate the agent
    evaluate_agent(model, env)
