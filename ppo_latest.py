import gym
from stable_baselines3 import PPO
from environment import GroceryStoreEnv  # Import your custom environment

# Define a custom grocery list
custom_grocery_list = ["Yogurt", "Bacon", "Potato Chips", "Pasta"]

# Initialize the environment with the custom grocery list
env = GroceryStoreEnv(grocery_list=custom_grocery_list)

# Check if you need to train the model
train_model = True  # Set this to True if you want to train; False to only load and test

if train_model:
    # Define and configure the PPO model with adjustments for better exploration
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="ppo_grocery_store_logs/",  # Directory for TensorBoard logs
        gamma=0.95,  # Lower gamma to focus on short-term rewards
        learning_rate=0.0003,  # Default learning rate
        batch_size=64,  # PPO batch size
        n_steps=2048,  # Number of steps to run for each environment in a rollout
        ent_coef=0.02  # Increased entropy coefficient to encourage more exploration
    )

    # Train the model
    model.learn(total_timesteps=10000000)
    # Save the trained model
    model.save("grocery_store_ppo_model_final")
else:
    # Load the trained model for testing
    model = PPO.load("grocery_store_ppo_model_final", env=env)

# Testing the trained model
num_episodes = 10
total_rewards = []
max_steps_per_episode = 500  # Added maximum step limit to avoid endless loops

for episode in range(num_episodes):
    observation = env.reset()[0]  # Only take the first element (the observation)
    done = False
    episode_reward = 0
    step_counter = 0

    while not done and step_counter < max_steps_per_episode:  # Limit each episode to max_steps_per_episode
        action, _states = model.predict(observation, deterministic=True)
        print(f"Episode {episode + 1}, Step {step_counter + 1}, Predicted action: {action}")
        observation, reward, done, truncated, info = env.step(action)
        env.render()
        episode_reward += reward
        step_counter += 1

    # Log the total reward for this episode
    total_rewards.append(episode_reward)
    print(f"Episode {episode + 1} Reward: {episode_reward}")

# Calculate and print the average reward across all episodes
avg_reward = sum(total_rewards) / num_episodes
print(f"Average Reward over {num_episodes} episodes: {avg_reward}")

# Close the environment
env.close()
