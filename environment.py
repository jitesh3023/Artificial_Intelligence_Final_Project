import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.image as mpimg

class GroceryStoreEnv(gym.Env):
    def __init__(self, grocery_list=None):
        super(GroceryStoreEnv, self).__init__()
        # Here I am defining actions_space. For now I am assuming our agent can
        # moving only in 4 directions that are - up, down, right, left. Diagonal
        # movements are restricted. Can be included later if needed.
        self.action_space = spaces.Discrete(4)
        # Observation space
        self.grid_size = (20, 20) 
        self.observation_space = spaces.Box(low=0, high=self.grid_size[0]-1, shape=(128,), dtype=np.int32)
        #self.observation_space = spaces.Box(low=0, high=self.grid_size[0]-1, shape=(8,), dtype=np.int32)
        self.visited_positions = set()

        # Defining Robot/Agent and item positions in the world
        self.entry_exit_position = np.array([0,0])
        self.robot_position = self.entry_exit_position.copy()
        self.done = False # Defining now, cause I think would be useful later for checking if the robot collected all the items. Could be useless

        self.aisles = [
            (3, 3), (4, 3), (5, 3), (6, 3), (7, 3), (8, 3), (9, 3), (10, 3), (11, 3), (12, 3), (13, 3), (14, 3), (15, 3), (16, 3),
            (3, 7), (4, 7), (5, 7), (6, 7), (7, 7), (8, 7), (9, 7), (10, 7), (11, 7), (12, 7), (13, 7), (14, 7), (15, 7), (16, 7),
            (3, 11), (4, 11), (5, 11), (6, 11), (7, 11), (8, 11), (9, 11), (10, 11), (11, 11), (12, 11), (13, 11), (14, 11), (15, 11), (16, 11),
            (3, 15), (4, 15), (5, 15), (6, 15), (7, 15), (8, 15), (9, 15), (10, 15), (11, 15), (12, 15), (13, 15), (14, 15), (15, 15), (16, 15)  
        ]

        self.items_list = {
            "milk": (0,3), "Eggs":(0,5), "Cheese":(0,7), "Yogurt":(0,9), "Cream":(0, 11), "Butter":(0,13), "Ice Cream":(0,15),
            "Potatoes": (3, 4), "Onions": (5,4), "Tomatoes":(7,4), "Lettuce":(9,4), "Carrot":(11,4), "Pepper":(13,4), "Cucumbers":(15,4), "Celery":(15, 6), "Broccoli":(13,6), "Mushrooms":(11,6), "Spinach":(9,6), "Corn":(7,6), "Cauliflower":(5,6), "Garlic":(3,6),
            "Banana":(3,8), "Berries":(5,8), "Apple":(7,8), "Grapes":(9,8), "Melons":(11,8), "Avocados":(13,8), "Mandarins":(15,8), "Oranges":(15,10), "Peaches":(13,10), "Pineapple":(11,10), "Cherries":(9,10), "Lemons":(7,10), "Kiwis":(5,10), "Mangoes":(3,10),
            "Baked Beans":(3,12), "Black Beans":(5,12), "Cookies":(7,12), "Crackers":(9,12), "Dried Fruits":(11,12), "Gelatin":(13,12), "Granola Bars":(15,12), "Nuts":(15,14), "Popcorn":(13,14), "Potato Chips":(11,14), "Pudding":(9,14), "Raisins":(7,14), "Pasta":(5,14), "Peanut Butter":(3,14),
            "Chicken":(4,19), "Lamb":(6,19), "Bacon":(8,19), "Ham":(10,19), "Turkey":(12,19), "Pork":(14,19), "Sausage":(16,19),           
            "Aluminum Foil":(19,3), "Garbage Bags":(19,5), "Napkins":(19,7), "Paper Plates":(19,9), "Plastics Bags":(19,11), "Straws":(19, 13), "Dish Soap":(19,15)
        }

        self.item_images = {
            "milk": mpimg.imread('Images/Milk.webp'),
            "Eggs": mpimg.imread('Images/Eggs.webp'),
            "Cheese": mpimg.imread('Images/Cheese.webp'),
            "Yogurt": mpimg.imread('Images/Yogurt.jpg'),
            "Cream": mpimg.imread('Images/Cream.webp'),
            "Butter": mpimg.imread('Images/Butter.jpeg'),
            "Ice Cream": mpimg.imread('Images/IceCream.jpeg'),
            "Potatoes": mpimg.imread('Images/Potatoes.jpeg'),
            "Onions": mpimg.imread('Images/Onions.jpeg'),
            "Tomatoes": mpimg.imread('Images/Tomatoes.jpeg'),
            "Lettuce": mpimg.imread('Images/Lettuce.jpeg'),
            "Carrot": mpimg.imread('Images/carrot.jpeg'),
            "Pepper": mpimg.imread('Images/pepper.jpeg'),
            "Cucumbers": mpimg.imread('Images/cucumbers.jpeg'),
            "Celery": mpimg.imread('Images/celery.jpeg'),
            "Broccoli": mpimg.imread('Images/broccoli.jpeg'),
            "Mushrooms": mpimg.imread('Images/mushroom.webp'),
            "Spinach": mpimg.imread('Images/spinach.jpeg'),
            "Corn": mpimg.imread('Images/corn.jpeg'),
            "Cauliflower": mpimg.imread('Images/cauliflower.jpeg'),
            "Garlic": mpimg.imread('Images/garlic.webp'),
            "Banana": mpimg.imread('Images/banana.jpeg'),
            "Berries": mpimg.imread('Images/berries.jpeg'),
            "Apple": mpimg.imread('Images/apple.jpeg'),
            "Grapes": mpimg.imread('Images/grapes.jpeg'),
            "Melons": mpimg.imread('Images/melons.jpeg'),
            "Avocados": mpimg.imread('Images/avocados.jpeg'),
            "Mandarins": mpimg.imread('Images/mandarines.jpeg'),
            "Oranges": mpimg.imread('Images/oranges.jpeg'),
            "Peaches": mpimg.imread('Images/peaches.jpeg'),
            "Pineapple": mpimg.imread('Images/pineapple.jpeg'),
            "Cherries": mpimg.imread('Images/cherries.jpeg'),
            "Lemons": mpimg.imread('Images/Lemon.jpeg'),
            "Kiwis": mpimg.imread('Images/kiwis.jpeg'),
            "Mangoes": mpimg.imread('Images/mangoes.jpeg'),
            "Baked Beans": mpimg.imread('Images/baked_beans.webp'),
            "Black Beans": mpimg.imread('Images/black_beans.jpeg'),
            "Cookies": mpimg.imread('Images/cookies.jpeg'),
            "Crackers": mpimg.imread('Images/crackers.jpeg'),
            "Dried Fruits": mpimg.imread('Images/dried_fruits.jpeg'),
            "Gelatin": mpimg.imread('Images/gelatin.jpeg'),
            "Granola Bars": mpimg.imread('Images/granola_bars.jpeg'),
            "Nuts": mpimg.imread('Images/nuts.jpeg'),
            "Popcorn": mpimg.imread('Images/popcorn.jpeg'),
            "Potato Chips": mpimg.imread('Images/potato_chips.jpeg'),
            "Pudding": mpimg.imread('Images/pudding.jpeg'),
            "Raisins": mpimg.imread('Images/raisins.jpeg'),
            "Pasta": mpimg.imread('Images/pasta.jpeg'),
            "Peanut Butter": mpimg.imread('Images/peanut_butter.jpeg'),
            "Chicken": mpimg.imread('Images/chicken.jpeg'),
            "Lamb": mpimg.imread('Images/lamb.jpeg'),
            "Bacon": mpimg.imread('Images/bacon.jpeg'),
            "Ham": mpimg.imread('Images/ham.jpeg'),
            "Turkey": mpimg.imread('Images/turkey.jpeg'),
            "Pork": mpimg.imread('Images/pork.jpeg'),
            "Sausage": mpimg.imread('Images/sausage.jpeg'),
            "Aluminum Foil": mpimg.imread('Images/aluminum_foil.jpeg'),
            "Garbage Bags": mpimg.imread('Images/garbage_bags.jpeg'),
            "Napkins": mpimg.imread('Images/napkins.jpeg'),
            "Paper Plates": mpimg.imread('Images/paper_plates.jpeg'),
            "Plastics Bags": mpimg.imread('Images/plastic_bags.jpeg'),
            "Straws": mpimg.imread('Images/straws.jpeg'),
            "Dish Soap": mpimg.imread('Images/dish_soap.jpeg')
        }

        self.original_grocery_list = grocery_list if grocery_list else ["milk", "Eggs", "Cheese"]
        self.grocery_list = self.original_grocery_list.copy()  
        self.last_position = tuple(self.robot_position)  # Track last position to penalize staying still
        self.max_steps_per_episode = 500  # Max steps per episode to avoid endless loops
        self.current_step = 0       
        self.stop_simulation = False
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
    
    def on_key_press(self, event):
        # Stop the simulation when 'q' is pressed
        if event.key == 'q':
            print("Key 'q' pressed. Stopping simulation.")
            self.stop_simulation = True

    # For returning robot and item's position at any instance
    def _get_observation(self):
        items_positions = list(self.items_list.values())
        return np.concatenate((self.robot_position, np.array(items_positions).flatten())).astype(np.int32)


    # For resetting the environment
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Check if the task was completed or if an explicit reset is required
        if not self.grocery_list:
            self.robot_position = self.entry_exit_position.copy()
            self.grocery_list = self.original_grocery_list.copy()
            print(f"Task completed! Resetting environment to start position: {self.robot_position}")
        else:
            print(f"Environment reset called, but task is incomplete. Maintaining current state.")
        self.visited_positions = set()
        self.current_step = 0
        self.last_position = tuple(self.robot_position)
        return self._get_observation(), {}

    def calculate_distance_to_items(self):
        # Only calculate distance if there are items left to collect
        if self.grocery_list:
            return min(np.linalg.norm(np.array(self.robot_position) - np.array(self.items_list[item])) 
                    for item in self.grocery_list)
        else:
            # Return a large value (or 0) if no items remain to avoid min() on an empty sequence
            return float('inf')
        

    def step(self, action=None):
        # Find the nearest item in the grocery list
        if not self.grocery_list:
            # No items left to collect
            return self._get_observation(), 0, True, False, {}

        nearest_item = min(
            self.grocery_list, key=lambda item: np.linalg.norm(self.robot_position - np.array(self.items_list[item]))
        )
        target_position = np.array(self.items_list[nearest_item])

        # Evaluate all possible moves (Up, Down, Left, Right)
        moves = {
            0: np.array([self.robot_position[0] - 1, self.robot_position[1]]),  # Up
            1: np.array([self.robot_position[0] + 1, self.robot_position[1]]),  # Down
            2: np.array([self.robot_position[0], self.robot_position[1] - 1]),  # Left
            3: np.array([self.robot_position[0], self.robot_position[1] + 1]),  # Right
        }

        # Filter out invalid moves (boundary or aisle constraints)
        valid_moves = {
            a: pos
            for a, pos in moves.items()
            if 0 <= pos[0] < self.grid_size[0] and 0 <= pos[1] < self.grid_size[1] and tuple(pos) not in self.aisles
        }

        # Select the move that minimizes the Euclidean distance to the target
        best_action = min(
            valid_moves.keys(),
            key=lambda a: np.linalg.norm(valid_moves[a] - target_position)
        )
        new_position = valid_moves[best_action]

        # Base penalty for every step
        reward = -1

        # Penalize revisiting positions
        if tuple(new_position) in self.visited_positions:
            reward -= 5  # Discourage revisiting

        # Add the position to visited
        self.visited_positions.add(tuple(new_position))

        # Calculate distance rewards
        old_distance = np.linalg.norm(self.robot_position - target_position)
        new_distance = np.linalg.norm(new_position - target_position)

        # Reward improvement in distance
        if new_distance < old_distance:
            reward += 20  # Increased bonus for moving closer to the target
        else:
            reward -= 5  # Penalty for moving away from the target

        # Update the robot's position
        self.robot_position = new_position

        # Check if the robot collects an item
        collected_items = []
        for item, pos in self.items_list.items():
            if item in self.grocery_list and np.array_equal(self.robot_position, np.array(pos)):
                reward += 100  # Large reward for collecting an item
                collected_items.append(item)

        # Remove collected items from the grocery list
        for item in collected_items:
            self.grocery_list.remove(item)

        # Check if the task is complete
        done = not self.grocery_list  # Task is done if all items are collected

        # Check for truncation (max steps per episode)
        self.current_step += 1
        truncated = self.current_step >= self.max_steps_per_episode

        # Log the robot's state for debugging
        print(f"Position: {self.robot_position}, Reward: {reward}, Remaining items: {self.grocery_list}")

        return self._get_observation(), reward, done, truncated, {}




    
    cmap = colors.ListedColormap(['white', 'green', 'blue', 'black']) # white for grid, green for robot, blue for items
    bounds = [0, 1, 2, 3, 4]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    def render(self, mode='human'):
        """Render the grocery store grid with images and visual elements."""
        self.ax.clear()
        grid = np.zeros(self.grid_size)

        # Mark aisles and items
        for aisle_pos in self.aisles:
            grid[aisle_pos] = 3  # Mark aisles with 3
        for item_pos in self.items_list.values():
            grid[item_pos] = 2  # Mark items with 2
        grid[tuple(self.robot_position)] = 1  # Mark robot with 1

        # Draw the grid
        self.ax.imshow(grid, cmap=self.cmap, norm=self.norm, zorder=1)

        # Draw images on top of the grid for all items
        for item, pos in self.items_list.items():
            image = self.item_images.get(item)
            if image is not None:  # Ensure the image exists
                self.ax.imshow(image, extent=(pos[1] - 0.5, pos[1] + 0.5, pos[0] - 0.5, pos[0] + 0.5),
                            aspect='auto', zorder=3)  # Higher z-order to overlay on the grid

        # Draw the robot
        self.ax.add_patch(plt.Rectangle((self.robot_position[1] - 0.5, self.robot_position[0] - 0.5),
                                        1, 1, color='green', zorder=4))  # Highest z-order

        # Set up gridlines
        self.ax.set_xticks(np.arange(-0.5, self.grid_size[1], 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, self.grid_size[0], 1), minor=True)
        self.ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Add grocery list and title
        self.ax.set_title('Grocery Store Environment')
        self.ax.text(-3, self.grid_size[0] - 1, "Grocery List:", fontsize=15, fontweight='bold', color='black')
        for idx, item in enumerate(self.grocery_list):
            self.ax.text(-3, self.grid_size[0] - (2 + idx), f"- {item}", fontsize=12, color='black')

        # Redraw the canvas
        self.fig.canvas.draw_idle()
        plt.pause(0.1)

    def close(self):
        pass

if __name__ == "__main__":
    # Create the environment
    env = GroceryStoreEnv(grocery_list=["milk", "Eggs", "Cheese"])

    # Reset the environment to get the initial observation
    observation, _ = env.reset()

    # Initialize variables for visualization
    done = False
    total_reward = 0

    # Run the simulation
    while not env.stop_simulation:  # Press 'q' to stop
        # Take a random action
        action = env.action_space.sample()

        # Step through the environment
        observation, reward, done, truncated, _ = env.step(action)

        # Render the environment
        env.render()

        # Print information for debugging
        print(f"Action: {action}, Reward: {reward}, Observation: {observation}, Done: {done}")

        # Accumulate the total reward
        total_reward += reward

        # If the episode is done, reset the environment
        if done or truncated:
            print("Episode finished! Resetting the environment...")
            observation, _ = env.reset()
            break

    print(f"Total Reward: {total_reward}")

    # Close the environment after finishing
    env.close()



# For testing the environment 

# env = GroceryStoreEnv()
# observation = env.reset()
# done = False
# total_reward = 0

# while not env.stop_simulation:
#     action = env.action_space.sample()
#     observation, reward, done, info = env.step(action)
#     env.render()
#     print(f"Action: {action}, Reward: {reward}, Observation: {observation}, Done: {done}")
#     total_reward += reward
#     if done:
#         print("Episode finished! Returning to entry/exit point.")
#         env.reset()
#         break 
# print("Total Reward:", total_reward)
# env.close()