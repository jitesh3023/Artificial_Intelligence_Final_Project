import gym
from gym import spaces
import numpy as np

class GroceryStoreEnv(gym.Env):
    def __init__(self):
        super(GroceryStoreEnv, self).__init__()
        # Here I am defining actions_space. For now I am assuming our agent can
        # moving only in 4 directions that are - up, down, right, left. Diagonal
        # movements are restricted. Can be included later if needed.
        self.action_space = spaces.Discrete(4)
        # Observation space
        self.grid_size = (10, 10) 
        self.observation_space = spaces.Box(low=0, high=self.grid_size[0]-1, shape=(8,), dtype=np.int32)
        
        # Defining Robot/Agent and item positions in the world
        self.entry_exit_position = np.array([0,0])
        self.robot_position = self.entry_exit_position.copy()
        self.done = False # Defining now, cause I think would be useful later for checking if the robot collected all the items. Could be useless

        self.aisles = {
            "vegetables": [(1,2), (1,3), (1,4)],
            "dairy": [(4, 2), (4,3), (4, 4)],
            "fruits": [(7,2), (7,3), (7,4)]
        }

        self.grocery_list = {
            "vegetables": (1, 2),
            "dairy": (4, 3),
            "fruits": (7, 4)
        }
    
    # For returning robot and item's position at any instance
    def _get_observation(self):
        items_positions = list(self.grocery_list.values())
        return np.concatenate((self.robot_position, np.array(items_positions).flatten()))

    # For resetting the environment
    def reset(self):
        self.robot_position = self.entry_exit_position.copy()
        self.done = False

        self.grocery_list = {
            "vegetables": (1, 2),
            "dairy": (4, 3),
            "fruits": (7, 4)
        }
        return self._get_observation()
    
    def step(self, action):
        if action == 0:  # up
            self.robot_position[0] = max(self.robot_position[0] - 1, 0)
        elif action == 1:  # down
            self.robot_position[0] = min(self.robot_position[0] + 1, self.grid_size[0] - 1)
        elif action == 2:  # left
            self.robot_position[1] = max(self.robot_position[1] - 1, 0)
        elif action == 3:  # right
            self.robot_position[1] = min(self.robot_position[1] + 1, self.grid_size[1] - 1)
        
        # Defining Rewards and Penalties
        reward = -1 
        collected_items = []

        for category, position in self.grocery_list.items():
            if np.array_equal(self.robot_position, np.array(position)):
                reward += 100 # For collecting an item
                collected_items.append(category)

        # After collecting the item that item should be removed from the list
        for item in collected_items:
            del self.grocery_list[item]

        # Condition to end the episode: all items collected and robot returned to entry/exit point
        if not self.grocery_list and np.array_equal(self.robot_position, self.entry_exit_position):
            reward += 200  # Extra reward for returning to the entry/exit point
            self.done = True

        return self._get_observation(), reward, self.done, {}
    
    def render(self, mode='human'):
        grid = np.zeros(self.grid_size)
        grid[tuple(self.robot_position)] = 1 # robot
        for item_pos in self.grocery_list.values():
            grid[tuple(item_pos)] = 2  # items
        print(grid)

    def close(self):
        pass


# For testing the environment 

env = GroceryStoreEnv()
observation = env.reset()
done = False
total_reward = 0

for i in range(50):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    env.render()
    print(f"Action: {action}, Reward: {reward}, Observation: {observation}, Done: {done}")
    total_reward += reward
    if done:
        print("Episode finished!")
        break
print("Total Reward:", total_reward)
env.close()
