import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, start=(0, 0), goal=(5, 5), grid_size=10):
        super(CustomEnv, self).__init__()
        self.grid_size = grid_size
        self.start = np.array(start, dtype=np.int32)
        self.goal = np.array(goal, dtype=np.int32)
        self.current_position = self.start.copy()
        
        # Action space: Up, Down, Left, Right
        self.action_space = spaces.Discrete(4)
        
        # Observation space: Position (x, y) on the grid
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size - 1, shape=(2,), dtype=np.int32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_position = self.start.copy()
        return self.current_position.astype(np.int32), {}

    def step(self, action):
        if action == 0:  # Up
            self.current_position[1] = min(self.current_position[1] + 1, self.grid_size - 1)
        elif action == 1:  # Down
            self.current_position[1] = max(self.current_position[1] - 1, 0)
        elif action == 2:  # Left
            self.current_position[0] = max(self.current_position[0] - 1, 0)
        elif action == 3:  # Right
            self.current_position[0] = min(self.current_position[0] + 1, self.grid_size - 1)

        if np.array_equal(self.current_position, self.goal):
            reward = 10
            terminated = True
        else:
            reward = -1
            terminated = False

        truncated = False
        return self.current_position.astype(np.int32), reward, terminated, truncated, {}

    def render(self, mode="rgb_array"):
        if mode == "human":
            grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
            grid[:] = "-"
            grid[self.goal[1], self.goal[0]] = "G"
            grid[self.current_position[1], self.current_position[0]] = "A"
            print("\n".join(["".join(row) for row in grid]))
        elif mode == "rgb_array":
            grid = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
            grid[:] = [255, 255, 255]  # White background
            grid[self.goal[1], self.goal[0]] = [0, 255, 0]  # Green for goal
            grid[self.current_position[1], self.current_position[0]] = [255, 0, 0]  # Red for agent
            return grid
