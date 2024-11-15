import time
import gymnasium as gym
from environment_V2 import GroceryStoreEnv

# Create the environment with a sample grocery list
grocery_list = ["apple", "banana", "carrot"]
env = GroceryStoreEnv(grid_size=5, grocery_list=grocery_list)

# Reset the environment
obs, _ = env.reset()
env.render()  # Initial rendering

# Run a few steps and render the environment after each step
for step in range(3):
    action = env.action_space.sample()  # Random action
    obs, reward, done, _, _ = env.step(action)
    print(f"Step: {step + 1}, Action: {action}, Reward: {reward}, Done: {done}")
    env.render()  # Render the environment after each step
    time.sleep(1)  # Pause to see each render step

    if done:
        print("Resetting environment")
        obs, _ = env.reset()
        env.render()
