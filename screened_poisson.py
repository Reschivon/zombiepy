from time import sleep
from matplotlib import pyplot as plt
import matplotlib
import torch
import torch.nn.functional as F
import math
from scipy.special import kv, iv

# Set the dimension to 2D
d = 2

# Constants
MAX_WALK_STEPS = 32
EPSILON_SHELL = 1e-2

def clear_debug_vis():
    ax = plt.gca()
    for c in ax.collections:
            c.remove()
            
def sample_unit_sphere_uniform(n_samples):
    """
    Samples n_samples points uniformly from the unit circle (2D sphere).
    """
    angles = torch.rand(n_samples) * 2 * math.pi
    x = torch.cos(angles)
    y = torch.sin(angles)
    return torch.stack((x, y), dim=1)  # Shape: (n_samples, 2)

class Env:
    def __init__(self, absorption_coeff, goal):
        self.absorption_coeff = absorption_coeff
        self.goal = goal
        
        # Define the obstacles as 2D torch tensors
        self.obstacles = [
            torch.tensor([2.0, 2.0]),
            torch.tensor([-2.0, -1.5]),
            torch.tensor([0.0, -2.5])
        ]
        self.obstacle_radius = 1.0  # Set the radius for each obstacle to 1.0 units
        
        self.domain_radius = 10.0

    def compute_distance_to_absorbing_boundary(self, current_pt):
        """
        Computes the distance to the absorbing boundary, considering both the domain
        boundary and the obstacles.
        The domain is assumed to be a circle with radius 10 centered at the origin.
        
        Parameters:
        - current_pt: A tensor of shape (n_walks, 2) representing the current points.
        
        Returns:
        - distances: A tensor of shape (n_walks,) representing the distance to the absorbing boundary.
        """
        # Compute the distance to the circular domain boundary (radius 10)
        dist_to_domain_boundary = self.domain_radius - current_pt.norm(dim=1)

        # Compute the distance to the nearest obstacle
        dist_to_obstacles = torch.full_like(dist_to_domain_boundary, float('inf'))  # Start with large values
        for obstacle in self.obstacles:
            dist_to_obstacle = (current_pt - obstacle).norm(dim=1) - self.obstacle_radius
            dist_to_obstacles = torch.minimum(dist_to_obstacles, dist_to_obstacle)

        # The absorbing boundary is either the domain boundary or the nearest obstacle
        return torch.minimum(dist_to_domain_boundary, dist_to_obstacles)

    def outside_bounding_domain(self, current_pt):
        """
        Checks if the points are outside the domain boundary (a circle with radius 10).
        
        Parameters:
        - current_pt: A tensor of shape (n_walks, 2) representing the current points.
        
        Returns:
        - A boolean tensor of shape (n_walks,) where True indicates the point is outside the boundary.
        """
        return current_pt.norm(dim=1) > self.domain_radius
    
    def source(self, pts):
        """
        Defines the source function at given points.
        """
        return torch.zeros(pts.size(0))
    
    def _hit_goal(self, pts, radii):
        free_dist = radii
        goal_dist = (pts - self.goal).pow(2).sum(1).sqrt()
        
        return free_dist > goal_dist
    
    def dirichlet(self, pts, radii):
        """
        Defines the Dirichlet boundary condition at given points.
        For simplicity, assume boundary condition is zero.
        """
        return torch.where(self._hit_goal(pts, radii), 1.0, 0.0)
    
# Define the Greens function for the Screened Poisson equation in 2D
class YukawaGreensFnBall2D:
    def __init__(self, lambda_, r_clamp=1e-4):
        self.lambda_ = lambda_
        self.sqrt_lambda = math.sqrt(lambda_)
        self.r_clamp = r_clamp
        self.c = None  # Center of the ball, shape: (n_walks, 2)
        self.R = None  # Radius of the ball, shape: (n_walks,)
        self.muR = None  # muR = R * sqrt_lambda, shape: (n_walks,)
        self.K0muR = None
        self.I0muR = None
        self.K1muR = None
        self.I1muR = None

    def update_ball(self, c, R):
        """
        Updates the ball center and radius.

        Parameters:
        - c: Tensor of shape (n_walks, 2)
        - R: Tensor of shape (n_walks,)
        """
        self.c = c
        self.R = R
        self.muR = R * self.sqrt_lambda  # Shape: (n_walks,)
        # Avoid zero to prevent singularities in Bessel functions
        self.muR = torch.clamp(self.muR, min=1e-10)
        self.K0muR = torch.from_numpy(kv(0, self.muR.cpu().numpy())).to(self.muR.device)  # Shape: (n_walks,)
        self.I0muR = torch.from_numpy(iv(0, self.muR.cpu().numpy())).to(self.muR.device)
        self.K1muR = torch.from_numpy(kv(1, self.muR.cpu().numpy())).to(self.muR.device)
        self.I1muR = torch.from_numpy(iv(1, self.muR.cpu().numpy())).to(self.muR.device)

    def sample_volume(self, n_walks):
        """
        Samples points inside the ball.

        Parameters:
        - n_walks: Number of walks (samples)

        Returns:
        - y_vol: Sampled points inside the ball, shape: (n_walks, 2)
        """
        # Uniformly sample radii using inverse transform sampling
        u = torch.rand(n_walks, 1)
        r = u ** (1.0 / d) * self.R.unsqueeze(1)  # Shape: (n_walks, 1)
        # Sample directions
        dir = sample_unit_sphere_uniform(n_walks)  # Shape: (n_walks, 2)
        # Compute points
        y_vol = self.c + r * dir  # Shape: (n_walks, 2)
        return y_vol

    def norm(self):
        """
        Computes the norm of the Greens function.

        Returns:
        - norm: Tensor of shape (n_walks,)
        """
        poisson_kernel = self.poisson_kernel()  # Shape: (n_walks,)
        return (1.0 - 2.0 * math.pi * poisson_kernel) / self.lambda_

    def poisson_kernel(self):
        """
        Evaluates the Poisson kernel at the boundary of the ball.

        Returns:
        - poisson_kernel: Tensor of shape (n_walks,)
        """
        I0muR = self.I0muR  # Shape: (n_walks,)
        return 1.0 / (2.0 * math.pi * I0muR)

    def poisson_kernel_dampening_factor(self, r):
        """
        Evaluates the radial dampening factor associated with the centered Poisson Kernel.

        Parameters:
        - r: Tensor of shape (n_walks,)

        Returns:
        - dampening_factor: Tensor of shape (n_walks,)
        """
        mur = r * self.sqrt_lambda  # Shape: (n_walks,)
        # Avoid zero to prevent singularities
        mur = torch.clamp(mur, min=1e-10)
        K1mur = torch.from_numpy(kv(1, mur.cpu().numpy())).to(mur.device)
        I1mur = torch.from_numpy(iv(1, mur.cpu().numpy())).to(mur.device)
        Q = K1mur + I1mur * (self.K0muR / self.I0muR)
        return mur * Q  # Shape: (n_walks,)

    def gradient(self, r, y_vol):
        """
        Evaluates the gradient of the Greens function.

        Parameters:
        - r: Tensor of shape (n_walks,)
        - y_vol: Points where the gradient is evaluated, shape: (n_walks, 2)

        Returns:
        - gradient: Tensor of shape (n_walks, 2)
        """
        mur = r * self.sqrt_lambda  # Shape: (n_walks,)
        K1mur = torch.from_numpy(kv(1, mur.cpu().numpy())).to(mur.device)
        I1mur = torch.from_numpy(iv(1, mur.cpu().numpy())).to(mur.device)
        Qr = self.sqrt_lambda * (K1mur - I1mur * (self.K1muR / self.I1muR))  # Shape: (n_walks,)
        gradient_norm = Qr / (2.0 * math.pi * r)  # Shape: (n_walks,)
        d = y_vol - self.c  # Shape: (n_walks, 2)
        return gradient_norm.unsqueeze(1) * d  # Shape: (n_walks, 2)

    def poisson_kernel_gradient(self):
        """
        Evaluates the gradient of the Poisson Kernel.

        Returns:
        - gradient: Tensor of shape (n_walks, 2)
        """
        
        d = self.y_surf - self.c  # Shape: (n_walks, 2)
        QR = self.sqrt_lambda / (self.R * self.I1muR)  # Shape: (n_walks,)
        gradient = QR.unsqueeze(1) / (2.0 * math.pi) * d  # Shape: (n_walks, 2)
        return gradient

    def sample_surface(self, n_walks):
        """
        Samples points on the surface of the ball.

        Parameters:
        - n_walks: Number of walks (samples)

        Returns:
        - y_surf: Sampled points on the surface of the ball, shape: (n_walks, 2)
        """
        directions = sample_unit_sphere_uniform(n_walks)  # Shape: (n_walks, 2)
        self.y_surf = self.c + self.R.unsqueeze(1) * directions  # Shape: (n_walks, 2)
        return self.y_surf

# Define the WalkState class
class WalkState:
    def __init__(self, current_pt, throughput):
        self.current_pt = current_pt  # Current positions, shape: (n_walks, 2)
        self.throughput = throughput  # Throughput, shape: (n_walks,)
        self.total_source_contribution = torch.zeros_like(throughput)
        self.greens_fn = None
        self.walk_length = torch.zeros_like(throughput, dtype=torch.int32)  # Shape: (n_walks,)
        
        n_walks = current_pt.size(0)
        self.walk_done = torch.full((n_walks,), False, dtype=torch.bool)

# Implement the walk function
def walk(dist_to_absorbing_boundary, state, vis):
    
    dist = dist_to_absorbing_boundary.clone()
    max_steps = MAX_WALK_STEPS

    for step in range(max_steps):
        # Update the Greens function ball
        state.greens_fn.update_ball(state.current_pt, dist)

        # Sample directions
        directions = sample_unit_sphere_uniform(torch.sum(~state.walk_done))  # Shape: (n_walks, 2)

        # Update positions
        state.current_pt[~state.walk_done] += dist.unsqueeze(1)[~state.walk_done] * directions  # Shape: (n_walks, 2)
                
        # Check for walks that have escaped the domain
        escaped = env.outside_bounding_domain(state.current_pt)
        state.walk_done = state.walk_done | escaped
        
        # Increment walk length
        state.walk_length += (~state.walk_done).int()

        # Compute the distance to the absorbing boundary for the next step
        dist = env.compute_distance_to_absorbing_boundary(state.current_pt)
        state.radii = dist

        # Check if walks have reached the epsilon shell
        reached_epsilon_shell = dist < EPSILON_SHELL
        state.walk_done = state.walk_done | reached_epsilon_shell
        
        # draw
        if vis:
            patches = [plt.Circle(center, size, color='green', fill=False, linestyle='solid', alpha=0.3) 
                    for center, size in zip(state.current_pt[~state.walk_done], 
                                                dist[~state.walk_done])]

            ax = plt.gca()

            coll = matplotlib.collections.PatchCollection(patches, match_original=True)
            ax.add_collection(coll)
            plt.pause(0.001)
        
        state.walk_done = state.walk_done | env._hit_goal(state.current_pt, dist)
        
        # Break if all walks are done
        if state.walk_done.all():
            print('All walks termianted')
            break

    # Return the final state
    return state

# Implement the estimateSolutionAndGrad function
def estimate_solution_and_grad(env: Env, n_walks, sample_pt, vis=False):
    """
    Estimates the solution and gradient at sample_pt using n_walks walks.

    Parameters:
    - pde: An instance of PDE
    - n_walks: Number of walks (samples)
    - sample_pt: Tensor of shape (2,)

    Returns:
    - estimated_solution: Estimated solution at sample_pt (scalar)
    - estimated_derivative: Estimated directional derivative at sample_pt (scalar)
    """
    # Initialize the starting points
    current_pt = sample_pt.unsqueeze(0).expand(n_walks, d).clone()  # Shape: (n_walks, 2)
    throughput = torch.ones(n_walks)  # Shape: (n_walks,)

    # Initialize the state
    state = WalkState(current_pt, throughput)
    state.greens_fn = YukawaGreensFnBall2D(env.absorption_coeff)

    # Compute the distance to the absorbing boundary
    dist_to_absorbing_boundary = env.compute_distance_to_absorbing_boundary(state.current_pt)
    dist_to_absorbing_boundary = torch.clamp(dist_to_absorbing_boundary, min=EPSILON_SHELL)

    # Set the first sphere radius
    first_sphere_radius = dist_to_absorbing_boundary * 0.99  # Slightly less than the distance

    # Update the Greens function with the first sphere
    state.greens_fn.update_ball(state.current_pt, first_sphere_radius)

    # Sample a point inside the ball for source contribution
    y_vol = state.greens_fn.sample_volume(n_walks)  # Shape: (n_walks, 2)
    r_vol = (y_vol - state.greens_fn.c).norm(dim=1)  # Shape: (n_walks,)

    # Compute the Greens function norm
    greens_fn_norm = state.greens_fn.norm()  # Shape: (n_walks,)

    # Compute the source contribution
    source_contribution = greens_fn_norm * env.source(y_vol)  # Shape: (n_walks,)
    state.total_source_contribution += state.throughput * source_contribution

    # Compute the gradient direction for the source contribution
    source_gradient_direction = state.greens_fn.gradient(r_vol, y_vol) / greens_fn_norm.unsqueeze(1)  # Shape: (n_walks, 2)

    # Sample a point on the surface of the ball
    y_surf = state.greens_fn.sample_surface(n_walks)  # Shape: (n_walks, 2)
    state.current_pt = y_surf  # Update the current position to the surface point

    # Update the throughput with the Poisson kernel
    poisson_kernel = state.greens_fn.poisson_kernel()  # Shape: (n_walks,)
    pdf_boundary = 1.0 / (2.0 * math.pi)  # Uniform sampling on circle
    state.throughput *= poisson_kernel / pdf_boundary
    
    # Compute the gradient direction for the boundary contribution
    boundary_gradient_direction = state.greens_fn.poisson_kernel_gradient() / (pdf_boundary * state.throughput.unsqueeze(1))  # Shape: (n_walks, 2)
    
    # Perform the walk
    dist_to_absorbing_boundary = env.compute_distance_to_absorbing_boundary(state.current_pt)
    dist_to_absorbing_boundary = torch.clamp(dist_to_absorbing_boundary, min=EPSILON_SHELL)
    state = walk(dist_to_absorbing_boundary, state, vis)

    # Compute the terminal contribution
    # For walks that reached the absorbing boundary, we use the Dirichlet condition
    # For simplicity, assume Dirichlet condition is zero    
    terminal_contribution = env.dirichlet(state.current_pt, state.radii)  # Shape: (n_walks,)
    # FIXME total_contribution = state.throughput * terminal_contribution + state.total_source_contribution  # Shape: (n_walks,)
    total_contribution = terminal_contribution
    
    # Compute the derivative
    # Direction for derivative (unit vector)
    direction_for_derivative = torch.tensor([1.0, 0.0])  # Along x-axis
    direction_for_derivative /= direction_for_derivative.norm()  # Ensure it's a unit vector

    # Compute the gradient estimates
    boundary_gradient_estimate = (total_contribution.unsqueeze(1) - source_contribution.unsqueeze(1)) * boundary_gradient_direction  # Shape: (n_walks, 2)
    source_gradient_estimate = source_contribution.unsqueeze(1) * source_gradient_direction  # Shape: (n_walks, 2)

    # Compute the directional derivative
    derivative_estimate = boundary_gradient_estimate + source_gradient_estimate  # Shape: (n_walks, 2)
    derivative = derivative_estimate.matmul(direction_for_derivative)  # Shape: (n_walks,)

    # Average the contributions
    estimated_solution = total_contribution.mean()
    estimated_derivative = derivative.mean()

    return estimated_solution.item(), derivative_estimate.mean(axis=0) #  estimated_derivative.item()

def visualize_init(env: Env, start):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    
    # Plot start end
    circle = plt.Circle(
        start.flatten().tolist(), 0.2, color='blue', alpha=0.5, fill=True
    )
    ax.add_artist(circle)
    
    circle = plt.Circle(
        env.goal.flatten().tolist(), 0.2, color='blue', alpha=0.5, fill=True
    )
    ax.add_artist(circle)
    
    # Plot obstacles
    for obs in env.obstacles:
        circle = plt.Circle(
            obs, 1.0, color='red', alpha=0.5
        )
        ax.add_artist(circle)
        
    # Plot domain boundary
    circle = plt.Circle(
        (0, 0), env.domain_radius, color='black', alpha=1, fill=False
    )
    ax.add_artist(circle)
    
    ax.set_xlim(-env.domain_radius - 1, env.domain_radius + 1)
    ax.set_ylim(-env.domain_radius - 1, env.domain_radius + 1)
    ax.set_aspect('equal', 'box')
    plt.title(f"PDE, screening {absorption_coeff}, walks {n_walks}")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    
    plt.pause(0.001)
    
# Example usage
if __name__ == "__main__":
    
    absorption_coeff = 100 
    
    n_walks = 10000 
    
    vis = False

    start = torch.tensor([-4., -4.]) 
    
    goal = torch.tensor([[4., 4.]])
        
    env = Env(absorption_coeff, goal)
    
    # Set up matplotlib
    visualize_init(env, start)
        
    # Gradient descent on start
    for _ in range(100):
        clear_debug_vis()
        
        # Plot start
        circle = plt.Circle(
            start.flatten().tolist(), 0.2, color='blue', alpha=0.5, fill=True
        )
        plt.gca().add_artist(circle)
    
        estimated_solution, estimated_derivative = estimate_solution_and_grad(env, n_walks, start, vis=vis)
        start += 0.2 * estimated_derivative / torch.norm(estimated_derivative)
        print('Derivative', estimated_derivative)
        
        plt.pause(0.001)
        

    print(f"Estimated solution at point {start.tolist()}: {estimated_solution}")
    print(f"Estimated directional derivative at point {start.tolist()}: {estimated_derivative}")
    
    plt.show()
