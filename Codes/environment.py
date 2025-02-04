import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.image as mpimg
from queue import PriorityQueue

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
        #self.aisles=[] 


        self.items_list = {
            "milk": (0,3), "Eggs":(0,5), "Cheese":(0,7), "Yogurt":(0,9), "Cream":(0, 11), "Butter":(0,13), "Ice Cream":(0,15),
            "Potatoes": (3, 4), "Onions": (5,4), "Tomatoes":(7,4), "Lettuce":(9,4), "Carrot":(11,4), "Pepper":(13,4), "Cucumbers":(15,4), "Celery":(15, 6), "Broccoli":(13,6), "Mushrooms":(11,6), "Spinach":(9,6), "Corn":(7,6), "Cauliflower":(5,6), "Garlic":(3,6),
            "Banana":(3,8), "Berries":(5,8), "Apple":(7,8), "Grapes":(9,8), "Melons":(11,8), "Avocados":(13,8), "Mandarins":(15,8), "Oranges":(15,10), "Peaches":(13,10), "Pineapple":(11,10), "Cherries":(9,10), "Lemons":(7,10), "Kiwis":(5,10), "Mangoes":(3,10),
            "Baked Beans":(3,12), "Black Beans":(5,12), "Cookies":(7,12), "Crackers":(9,12), "Dried Fruits":(11,12), "Gelatin":(13,12), "Granola Bars":(15,12), "Nuts":(15,14), "Popcorn":(13,14), "Potato Chips":(11,14), "Pudding":(9,14), "Raisins":(7,14), "Pasta":(5,14), "Peanut Butter":(3,14),
            "Chicken":(4,19), "Lamb":(6,19), "Bacon":(8,19), "Ham":(10,19), "Turkey":(12,19), "Pork":(14,19), "Sausage":(16,19),           
            "Aluminum Foil":(19,3), "Garbage Bags":(19,5), "Napkins":(19,7), "Paper Plates":(19,9), "Plastics Bags":(19,11), "Straws":(19, 13), "Dish Soap":(19,15)
        }

        self.item_images = {
            "milk": mpimg.imread('../Grocery_Store_Items_Images/Milk.webp'),
            "Eggs": mpimg.imread('../Grocery_Store_Items_Images/Eggs.webp'),
            "Cheese": mpimg.imread('../Grocery_Store_Items_Images/Cheese.webp'),
            "Yogurt": mpimg.imread('../Grocery_Store_Items_Images/Yogurt.jpg'),
            "Cream": mpimg.imread('../Grocery_Store_Items_Images/Cream.webp'),
            "Butter": mpimg.imread('../Grocery_Store_Items_Images/Butter.jpeg'),
            "Ice Cream": mpimg.imread('../Grocery_Store_Items_Images/IceCream.jpeg'),
            "Potatoes": mpimg.imread('../Grocery_Store_Items_Images/Potatoes.jpeg'),
            "Onions": mpimg.imread('../Grocery_Store_Items_Images/Onions.jpeg'),
            "Tomatoes": mpimg.imread('../Grocery_Store_Items_Images/Tomatoes.jpeg'),
            "Lettuce": mpimg.imread('../Grocery_Store_Items_Images/Lettuce.jpeg'),
            "Carrot": mpimg.imread('../Grocery_Store_Items_Images/carrot.jpeg'),
            "Pepper": mpimg.imread('../Grocery_Store_Items_Images/pepper.jpeg'),
            "Cucumbers": mpimg.imread('../Grocery_Store_Items_Images/cucumbers.jpeg'),
            "Celery": mpimg.imread('../Grocery_Store_Items_Images/celery.jpeg'),
            "Broccoli": mpimg.imread('../Grocery_Store_Items_Images/broccoli.jpeg'),
            "Mushrooms": mpimg.imread('../Grocery_Store_Items_Images/mushroom.webp'),
            "Spinach": mpimg.imread('../Grocery_Store_Items_Images/spinach.jpeg'),
            "Corn": mpimg.imread('../Grocery_Store_Items_Images/corn.jpeg'),
            "Cauliflower": mpimg.imread('../Grocery_Store_Items_Images/cauliflower.jpeg'),
            "Garlic": mpimg.imread('../Grocery_Store_Items_Images/garlic.webp'),
            "Banana": mpimg.imread('../Grocery_Store_Items_Images/banana.jpeg'),
            "Berries": mpimg.imread('../Grocery_Store_Items_Images/berries.jpeg'),
            "Apple": mpimg.imread('../Grocery_Store_Items_Images/apple.jpeg'),
            "Grapes": mpimg.imread('../Grocery_Store_Items_Images/grapes.jpeg'),
            "Melons": mpimg.imread('../Grocery_Store_Items_Images/melons.jpeg'),
            "Avocados": mpimg.imread('../Grocery_Store_Items_Images/avocados.jpeg'),
            "Mandarins": mpimg.imread('../Grocery_Store_Items_Images/mandarines.jpeg'),
            "Oranges": mpimg.imread('../Grocery_Store_Items_Images/oranges.jpeg'),
            "Peaches": mpimg.imread('../Grocery_Store_Items_Images/peaches.jpeg'),
            "Pineapple": mpimg.imread('../Grocery_Store_Items_Images/pineapple.jpeg'),
            "Cherries": mpimg.imread('../Grocery_Store_Items_Images/cherries.jpeg'),
            "Lemons": mpimg.imread('../Grocery_Store_Items_Images/Lemon.jpeg'),
            "Kiwis": mpimg.imread('../Grocery_Store_Items_Images/kiwis.jpeg'),
            "Mangoes": mpimg.imread('../Grocery_Store_Items_Images/mangoes.jpeg'),
            "Baked Beans": mpimg.imread('../Grocery_Store_Items_Images/baked_beans.webp'),
            "Black Beans": mpimg.imread('../Grocery_Store_Items_Images/black_beans.jpeg'),
            "Cookies": mpimg.imread('../Grocery_Store_Items_Images/cookies.jpeg'),
            "Crackers": mpimg.imread('../Grocery_Store_Items_Images/crackers.jpeg'),
            "Dried Fruits": mpimg.imread('../Grocery_Store_Items_Images/dried_fruits.jpeg'),
            "Gelatin": mpimg.imread('../Grocery_Store_Items_Images/gelatin.jpeg'),
            "Granola Bars": mpimg.imread('../Grocery_Store_Items_Images/granola_bars.jpeg'),
            "Nuts": mpimg.imread('../Grocery_Store_Items_Images/nuts.jpeg'),
            "Popcorn": mpimg.imread('../Grocery_Store_Items_Images/popcorn.jpeg'),
            "Potato Chips": mpimg.imread('../Grocery_Store_Items_Images/potato_chips.jpeg'),
            "Pudding": mpimg.imread('../Grocery_Store_Items_Images/pudding.jpeg'),
            "Raisins": mpimg.imread('../Grocery_Store_Items_Images/raisins.jpeg'),
            "Pasta": mpimg.imread('../Grocery_Store_Items_Images/pasta.jpeg'),
            "Peanut Butter": mpimg.imread('../Grocery_Store_Items_Images/peanut_butter.jpeg'),
            "Chicken": mpimg.imread('../Grocery_Store_Items_Images/chicken.jpeg'),
            "Lamb": mpimg.imread('../Grocery_Store_Items_Images/lamb.jpeg'),
            "Bacon": mpimg.imread('../Grocery_Store_Items_Images/bacon.jpeg'),
            "Ham": mpimg.imread('../Grocery_Store_Items_Images/ham.jpeg'),
            "Turkey": mpimg.imread('../Grocery_Store_Items_Images/turkey.jpeg'),
            "Pork": mpimg.imread('../Grocery_Store_Items_Images/pork.jpeg'),
            "Sausage": mpimg.imread('../Grocery_Store_Items_Images/sausage.jpeg'),
            "Aluminum Foil": mpimg.imread('../Grocery_Store_Items_Images/aluminum_foil.jpeg'),
            "Garbage Bags": mpimg.imread('../Grocery_Store_Items_Images/garbage_bags.jpeg'),
            "Napkins": mpimg.imread('../Grocery_Store_Items_Images/napkins.jpeg'),
            "Paper Plates": mpimg.imread('../Grocery_Store_Items_Images/paper_plates.jpeg'),
            "Plastics Bags": mpimg.imread('../Grocery_Store_Items_Images/plastic_bags.jpeg'),
            "Straws": mpimg.imread('../Grocery_Store_Items_Images/straws.jpeg'),
            "Dish Soap": mpimg.imread('../Grocery_Store_Items_Images/dish_soap.jpeg')
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
        

    def find_shortest_path(self, start, target):
        """Find the shortest path using A* algorithm."""
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        queue = PriorityQueue()
        queue.put((0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        while not queue.empty():
            _, current = queue.get()
            
            if current == target:
                # Reconstruct the path
                path = []
                while current:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path  # Return the shortest path
            
            for direction in directions:
                neighbor = (current[0] + direction[0], current[1] + direction[1])
                # Check if the neighbor is valid
                if (0 <= neighbor[0] < self.grid_size[0] and
                    0 <= neighbor[1] < self.grid_size[1] and
                    neighbor not in self.aisles):
                    new_cost = cost_so_far[current] + 1  # Uniform cost
                    if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                        cost_so_far[neighbor] = new_cost
                        priority = new_cost + np.linalg.norm(np.array(neighbor) - np.array(target))  # Heuristic
                        queue.put((priority, neighbor))
                        came_from[neighbor] = current

        return None  # No path found


    def step(self, action=None):
        """Modified step function to use pathfinding for valid paths."""
        if not self.grocery_list:
            return self._get_observation(), 0, True, False, {}
        
        # Find the nearest item along a valid path
        shortest_path = None
        target_item = None
        for item in self.grocery_list:
            target_position = tuple(self.items_list[item])
            path = self.find_shortest_path(tuple(self.robot_position), target_position)
            if path:
                if shortest_path is None or len(path) < len(shortest_path):
                    shortest_path = path
                    target_item = item
        
        if not shortest_path:
            print("No valid path to any item. Ending episode.")
            return self._get_observation(), -1000, True, False, {}  # Penalize and end if stuck
        
        # Ensure there is a next position to move to
        if len(shortest_path) > 1:
            new_position = np.array(shortest_path[1])  # Move to the next step in the path
        else:
            new_position = np.array(shortest_path[0])  # Already at the target

        # Reward calculation
        reward = -1  # Base penalty for each step
        if tuple(new_position) in self.visited_positions:
            reward -= 5  # Penalize revisiting
        
        old_distance = np.linalg.norm(self.robot_position - np.array(self.items_list[target_item]))
        new_distance = np.linalg.norm(new_position - np.array(self.items_list[target_item]))
        
        if new_distance < old_distance:
            reward += 20  # Reward moving closer
        else:
            reward -= 5  # Penalize moving away

        # Update robot position and visited positions
        self.robot_position = new_position
        self.visited_positions.add(tuple(new_position))
        
        # Check if the robot collects the item
        collected_items = []
        for item, pos in self.items_list.items():
            if item in self.grocery_list and np.array_equal(self.robot_position, np.array(pos)):
                reward += 100  # Reward for collecting item
                collected_items.append(item)
        
        for item in collected_items:
            self.grocery_list.remove(item)
        
        # Check if the task is complete
        done = not self.grocery_list
        self.current_step += 1
        truncated = self.current_step >= self.max_steps_per_episode
        
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