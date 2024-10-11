import numpy as np
import matplotlib.pyplot as plt
import torch

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

global dolog

def log(*args, **kwargs):
    if dolog: print(*args, **kwargs)

class Env:
    def __init__(self):
        self.obstacles = []
        self.domain_min = np.array([-10, -10])
        self.domain_max =  np.array([10, 10])
    
    def sample_free(self):
        # Sample a random point in the configuration space
        new_position = np.random.uniform(self.domain_min, self.domain_max)

        if self.is_in_collision(new_position):
            return self.sample_free()
        
        sample = plt.Circle(
            new_position, 0.05, color='black', fill=False, alpha=0.5
        )
        plt.gca().add_artist(sample)
    
        return new_position
        
    def distance_to_boundary(self, x):
        # Compute the distance to the domain boundary
        
        d_boundary = np.min(np.concatenate([
                            np.abs(x - self.domain_min),
                            np.abs(x - self.domain_max)]))
        
        # Compute the distance to the obstacles
        d_obstacles = []
        for obs in self.obstacles:
            d = np.linalg.norm(x - obs['center']) - obs['radius']
            d_obstacles.append(d)
        d_obstacles = np.array(d_obstacles)
        # h(x) is the minimum of d_boundary and d_obstacles
        h_x = min(d_boundary, np.min(d_obstacles))
        return h_x

    def is_in_collision(self, x):
        # Check collision with domain boundary
        if np.any((x < self.domain_min) | (self.domain_max < x)):
            return True
        
        # Check collision with obstacles
        for obs in self.obstacles:
            if np.linalg.norm(x - obs['center']) <= obs['radius']:
                return True
        return False

class SphereNode:
    def __init__(self, position, parent, radius, cost):
        self.position = position
        self.parent = parent
        self.radius = radius  # Will be set later based on distance_to_boundary
        self.cost = cost  # Cost from start node to this node


def visualize_init(env: Env, start, goal):
    fig = plt.figure(1, figsize=(10, 10))
    ax = plt.gca()
    
    # PLot start end
    circle = plt.Circle(
        start, 0.2, color='blue', alpha=0.5, fill=True
    )
    ax.add_artist(circle)
    
    circle = plt.Circle(
        goal, 0.2, color='blue', alpha=0.5, fill=True
    )
    ax.add_artist(circle)
    
    # Plot obstacles
    for obs in env.obstacles:
        circle = plt.Circle(
            obs['center'], obs['radius'], color='red', alpha=0.5
        )
        ax.add_artist(circle)
        
    # Plot domain boundary
    boundary = plt.Rectangle(
        env.domain_min, 
        env.domain_max[0] - env.domain_min[0], 
        env.domain_max[1] - env.domain_min[1],
        color='black', fill=False
    )
    ax.add_artist(boundary)
    
    ax.set_xlim(env.domain_min[0] - 1, env.domain_max[0] + 1)
    ax.set_ylim(env.domain_min[1] - 1, env.domain_max[1] + 1)
    ax.set_aspect('equal', 'box')
    plt.title('RRT WoS Path Planning')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    
def visualize_path(path):
        
    # Plot path
    if path is not None:
        print("Plotting", path)
        plt.plot(path[:, 0], path[:, 1], '-o', color='blue', markersize=0.1)
        plt.pause(0.001)
    else:
        print("No path to visualize.")


def select_sphere_to_expand(rand_point, tree) -> SphereNode:
    # Find the closest sphere in the tree to the random point
    min_dist = float('inf')
    nearest_node = None
    for node in tree:
        dist = np.linalg.norm(node.position - rand_point)
        if dist < min_dist:
            min_dist = dist
            nearest_node = node
    return nearest_node

def get_expansion_sphere(nearest_node, env, eps) -> SphereNode:
    if nearest_node.radius <= eps:
        log(f"   get_expansion_sphere: Can't expand from here since radius is small {nearest_node.radius}")
        return None  # Cannot expand from this node

    # Sample a point on the surface of the sphere centered at nearest_node.position
    theta = np.random.uniform(0, 2 * np.pi)
    new_position = nearest_node.position + nearest_node.radius * np.array([np.cos(theta), np.sin(theta)])

    if env.is_in_collision(new_position):
        log(f"   get_expansion_sphere: Bad new position {new_position} (in collision) ")
        return None  # New position is in collision

    # Create new node
    new_node = SphereNode(new_position, 
                          parent=nearest_node, 
                          radius=env.distance_to_boundary(new_position),
                          cost=nearest_node.cost + np.linalg.norm(new_position - nearest_node.position))
    return new_node

def extract_path(goal_node):
    # Follow parent links from goal_node to start_node
    path = []
    node = goal_node
    while node is not None:
        path.append(node.position)
        node = node.parent
    path.reverse()  # Reverse to get path from start to goal
    return np.array(path)

def plan_path_wos(start, goal, env, max_iterations=5000, vis=False):
    eps = 0.1
        
    # Initialize tree with start node
    start_node = SphereNode(start, 
                            parent=None, 
                            radius=env.distance_to_boundary(start),
                            cost=0.0)
    
    tree = [start_node]
    goal_reached = False
    goal_node = None

    for iteration in range(max_iterations):
        
        log(f"Now iteration {iteration}")
        
        # Select sphere to expand
        nearest_node = select_sphere_to_expand(env.sample_free(), tree)
        if nearest_node is None:
            continue
        log(f"Selected sphere {nearest_node.position} to expand")

        # Get expansion sphere
        new_node = get_expansion_sphere(nearest_node, env, eps)
        if new_node is None:
            continue  # Skip if expansion failed
        log(f"Expand {nearest_node.position} --> {new_node.position}")

        tree.append(new_node)
        
        if vis:
            sphere = plt.Circle(
                new_node.position, new_node.radius, color='green', fill=False, linestyle='solid', alpha=0.3
            )
            plt.gca().add_artist(sphere)
            plt.pause(0.0001)

        # Check if goal is reached
        if np.linalg.norm(new_node.position - goal) <= new_node.radius:
            goal_reached = True
            dist_new_to_goal = np.linalg.norm(goal - new_node.position)
            goal_node = SphereNode(goal, new_node, radius=0, cost=dist_new_to_goal)
            print(f"Goal reached at iteration {iteration}")
            break

    if not goal_reached:
        print("Goal not reached within the maximum number of iterations.")
        return None

    # Extract path from goal to start
    path = extract_path(goal_node)
    return path

def purge_circles():
    ax = plt.gca()
    for c in ax.collections:
            c.remove()
            
if __name__ == "__main__":
    dolog = False
    
    env = Env()
    
    # Define environment
    env.obstacles = [
        {'center': np.array([2.0, 2.0]), 'radius': 1.0},
        {'center': np.array([-2.0, -1.5]), 'radius': 1.0},
        {'center': np.array([0.0, -2.5]), 'radius': 1.0},
    ]
    
    # Define domain boundaries
    env.domain_min = np.array([-10.0, -10.0])
    env.domain_max = np.array([10.0, 10.0])
    
    # Define start and goal positions
    start = np.array([-4.0, -4.0])
    goal = np.array([4.0, 4.0])
    

    # Execute WoS algorithm
    for _ in range(1000):
        plt.cla()
        visualize_init(env, start, goal)
        
        path = plan_path_wos(
            start, goal, env, max_iterations=10000, vis=True
        )
        
        # Visualize the path and safe spheres
        visualize_path(path)
        
        plt.pause(0.5)
    