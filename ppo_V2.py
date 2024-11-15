import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from environment_V2 import GroceryStoreEnv  # Replace with the correct path to your environment

# Define the grocery list
grocery_list = ["apple", "banana", "carrot"]

# Initialize the environment
env = GroceryStoreEnv(grid_size=5, grocery_list=grocery_list)

# Wrap the environment for training
vec_env = make_vec_env(lambda: env, n_envs=1)
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)  # Add normalization wrapper

# Initialize the PPO model
model = PPO("MlpPolicy", vec_env, verbose=1)

# Train the model
print("Training the model...")
model.learn(total_timesteps=100000)

# Save the trained model and normalization wrapper
model.save("ppo_grocery_store")
vec_env.save("ppo_grocery_store_norm.pkl")  # Save normalization separately
print("Model and normalization wrapper saved.")

# Load the trained model
print("Loading the trained model...")
vec_env = make_vec_env(lambda: env, n_envs=1)
vec_env = VecNormalize.load("ppo_grocery_store_norm.pkl", vec_env)  # Load normalization
model = PPO.load("ppo_grocery_store", env=vec_env)  # Load the trained model

# Testing the trained model
print("Testing the trained model...")
obs, _ = env.reset()  # Reset environment for testing
done = False

while not done:
    print(f"Observation: {obs}")  # Debug: Check the observation values
    action, _states = model.predict(obs, deterministic=True)  # Get action from the trained model
    obs, reward, done, _, _ = env.step(action)  # Take a step in the environment
    print(f"Action: {action}, Reward: {reward}, Done: {done}")  # Debug: Log action and reward
    env.render()  # Visualize the environment
