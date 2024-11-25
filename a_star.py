import numpy as np
import heapq
from environment import GroceryStoreEnv
import matplotlib.pyplot as plt 

class AStarSolver:
    def __init__(self, env, goal_positions):
        self.env = env
        self.goal_positions = goal_positions

    # Desining heuristic cost which is nothing but the manhattan distance between the 2 cells
    def heuristic(self, pos1, pos2):
        return abs(pos1[0]-pos2[0]) + abs(pos1[1]-pos2[1])
         
    # Main A star logic
    def astar(self, start, goal):
        open_list = []
        heapq.heappush(open_list, (0, start)) # {dist, node}
        came_from = {}
        g_score = {start: 0} # {node, weight}
        f_score = {start: self.heuristic(start, goal)} # {node, heuristic}

        while open_list:
            _, current = heapq.heappop(open_list) # obtaining the minimum cost node, which is the top node cause we are using minheap. first = dist, second = node
            if current == goal:
                return self.reconstruct_path(came_from, current)

            neighbors = self.get_neighbors(current)
            for neighbor in neighbors:
                tentative_g_score = g_score[current] + 1 # Here I am considering the weight or cost to take one step in any direction is 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score 
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f_score[neighbor], neighbor)) # dist, node
            
        return None
    
    def reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1] # reversing the path cause we want from start to goal
    
    def get_neighbors(self, pos):
        neighbors = []
        directions = [(-1,0), (1,0), (0,-1), (0,1)] # Moving up, down, left, right respectively
        for d in directions:
            neighbor = (pos[0] + d[0], pos[1] + d[1])
            if 0 <= neighbor[0] < self.env.grid_size[0] and 0 <= neighbor[1] < self.env.grid_size[1]:
                if neighbor not in self.env.aisles:  
                    neighbors.append(neighbor)
        return neighbors
    
    def solve(self):
        current_position = tuple(self.env.robot_position)
        total_path = []
        for item, goal_position in self.goal_positions.items():
            if item in self.env.grocery_list:
                print(f"Finding path to {item} at {goal_position}")
                path = self.astar(current_position, goal_position)
                if path:
                    total_path.extend(path)
                    current_position = goal_position 
                else:
                    print(f"Could not find path to {item}")

        # After collecting all items the robot shoudl return to the entry/exit point
        if current_position != tuple(self.env.entry_exit_position):
            print(f"Returning to the entry/exit point at {self.env.entry_exit_position}")
            return_path = self.astar(current_position, tuple(self.env.entry_exit_position))
            if return_path:
                total_path.extend(return_path) 
            else:
                print("Could not find path back to the entry/exit point")
        return total_path
    
if __name__ == "__main__":
    env = GroceryStoreEnv()
    env.reset()
    env.grocery_list = ["Butter", "Potatoes", "Crackers"]
    grocery_items = {item: env.items_list[item] for item in env.grocery_list}
    astar_solver = AStarSolver(env, grocery_items)
    solution_path = astar_solver.solve()
    if solution_path:
        print(f"Total solution path: {solution_path}")
        for step in solution_path:
            env.robot_position = np.array(step)
            #robot position is a global variable between a_star and env  
            env.render()  
            #plt.pause(0.1)  
    else:
        print("No path found to complete the grocery list.")
    env.close()

