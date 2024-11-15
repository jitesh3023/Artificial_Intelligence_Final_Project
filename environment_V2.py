import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import numpy as np

class GroceryStoreEnv(gym.Env):
    def __init__(self, grid_size=5, grocery_list=None):
        super(GroceryStoreEnv, self).__init__()
        
        self.grid_size = grid_size
        self.grocery_list = grocery_list if grocery_list else []
        self.agent_pos = [0, 0]
        self.items_pos = {}
        self.collected_items = []

        # Action space: 0 = up, 1 = down, 2 = left, 3 = right, 4 = pick
        self.action_space = spaces.Discrete(5)
        
        # Observation space
        self.observation_space = spaces.Box(
            low=0,
            high=self.grid_size - 1,
            shape=(2 + len(self.grocery_list) * 2 + 1,),  # Agent position + items positions + collected items count
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        # Reset the environment to its initial state
        self.agent_pos = [0, 0]
        self.collected_items = []

        # Predefined positions for all possible items
        self.all_items_pos = {
            "apple": [1, 2],
            "banana": [3, 4],
            "carrot": [0, 3],
            "milk": [2, 2],
            "bread": [4, 1],
            # Add more items and positions as needed
        }

        # Only items in the grocery list should be collected
        self.items_pos = {item: self.all_items_pos[item] for item in self.grocery_list}

        return self._get_observation(), {}

    def step(self, action):
        if action == 0 and self.agent_pos[1] > 0:  # Move up
            self.agent_pos[1] -= 1
        elif action == 1 and self.agent_pos[1] < self.grid_size - 1:  # Move down
            self.agent_pos[1] += 1
        elif action == 2 and self.agent_pos[0] > 0:  # Move left
            self.agent_pos[0] -= 1
        elif action == 3 and self.agent_pos[0] < self.grid_size - 1:  # Move right
            self.agent_pos[0] += 1
        elif action == 4:  # Pick up item
            for item, pos in self.items_pos.items():
                if pos == self.agent_pos:
                    self.collected_items.append(item)
                    del self.items_pos[item]
                    break

        reward = self._get_reward()
        done = self._is_done()
        return self._get_observation(), reward, done, False, {}

    def _get_observation(self):
        # Flatten items_pos into a fixed-length 1D vector
        items_positions = (
            np.array(list(self.items_pos.values()), dtype=np.float32)
            if self.items_pos else np.zeros((len(self.grocery_list), 2), dtype=np.float32)
        )
        if len(items_positions) < len(self.grocery_list):
            padding = np.zeros((len(self.grocery_list) - len(items_positions), 2), dtype=np.float32)
            items_positions = np.vstack((items_positions, padding))

        return np.concatenate([
            np.array(self.agent_pos, dtype=np.float32),  # Agent position
            items_positions.flatten(),  # Flattened items positions
            np.array([len(self.collected_items)], dtype=np.float32)  # Count of collected items
        ])

    def _get_reward(self):
        # Encourage collecting items and penalize idle steps
        if len(self.collected_items) == len(self.grocery_list):
            return 10  # Reward for collecting all items
        elif self.agent_pos in self.items_pos.values():
            return 1  # Small reward for reaching an item's position
        else:
            return -0.1  # Small penalty for each step

    def _is_done(self):
        return len(self.collected_items) == len(self.grocery_list) and self.agent_pos == [0, 0]

    def render(self, mode='human'):
        # Create a blank grid with all cells set to white
        grid = np.zeros((self.grid_size, self.grid_size))

        # Place all store items
        for i, (item, pos) in enumerate(self.all_items_pos.items(), start=1):
            grid[pos[1]][pos[0]] = i

        # Set the agent's position
        grid[self.agent_pos[1]][self.agent_pos[0]] = -1

        # Plot the grid
        plt.figure(figsize=(6, 6))
        plt.imshow(grid, cmap='Pastel1', origin='upper', extent=[0, self.grid_size, 0, self.grid_size])
        plt.xticks(range(self.grid_size))
        plt.yticks(range(self.grid_size))
        plt.grid(True, color='black', linewidth=1)

        # Annotate the grid
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if grid[j, i] == -1:
                    plt.text(i + 0.5, j + 0.5, 'R', ha='center', va='center', color='white', fontsize=16)
                elif grid[j, i] > 0:
                    item_name = list(self.all_items_pos.keys())[int(grid[j, i]) - 1]
                    if item_name in self.grocery_list:
                        plt.text(i + 0.5, j + 0.5, item_name, ha='center', va='center', color='black', fontsize=8)
                    else:
                        plt.text(i + 0.5, j + 0.5, item_name, ha='center', va='center', color='gray', fontsize=8)

        plt.title("Grocery Store Environment")
        plt.show()

