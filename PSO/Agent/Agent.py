import collections

import gymnasium as gym
import numpy as np

from PSO.Environment.ObjectiveFunctions.ObjectiveFunction import ObjectiveFunction

class Agent(gym.Env):

    def __init__(self,
                 num_particles=30,
                 max_steps=5000,
                 agent_step_size=10,
                 adaptive_nt=False,
                 nt_range=(1, 100),
                 ):
        self._gbest_history = None
        self.max_steps = max_steps
        self.current_step = 0
        self.last_gbest = float('inf')
        self.adaptive_nt = adaptive_nt
        self.nt_range = nt_range
        self._current_nt = agent_step_size if not self.adaptive_nt else self.nt_range[0]
        self.num_particles = num_particles

        # initialize swarm
        self.lower_bound = -1
        self.upper_bound = 1
        self.number_dimensions = 3
        self.positions = np.random.uniform(low=self.lower_bound, high=self.upper_bound,
                                           size=(self.num_particles, self.number_dimensions))
        self.velocities = np.zeros((self.num_particles, self.number_dimensions))
        self.previous_positions = self.positions.copy()
        self.pbest_positions = self.positions.copy()

        # pass initial for 1st G Best
        self.gbest_position = self.positions[0].copy() if self.num_particles > 0 else np.zeros(self.number_dimensions)
        self.gbest_value = np.inf

        # empty init environment
        self.pbest_values = None
        self.bounds = None
        self.objective_function = None

        # observation space
        obs_low = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=obs_low, high=obs_high, shape=(4,), dtype=np.float32
        )

    def load_environment(self,
                         objective_function: ObjectiveFunction):
        self.objective_function = objective_function
        self.number_dimensions = objective_function.dim
        self.bounds = objective_function.bounds
        self.lower_bound, self.upper_bound = objective_function.bounds
        self.positions = np.random.uniform(low=self.lower_bound, high=self.upper_bound,
                                           size=(self.num_particles, self.number_dimensions))
        self.velocities = np.zeros((self.num_particles, self.number_dimensions))

        # history
        self._gbest_history = collections.deque()

        # initialize step
        self.pbest_positions = self.positions.copy()
        self.pbest_values = self.objective_function.evaluate_matrix(self.positions)
        min_idx = np.argmin(self.pbest_values)

        # observation space reset
        obs_low = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=obs_low, high=obs_high, shape=(4,), dtype=np.float32
        )

        # find initial best
        if len(self.pbest_values) > 0 and np.isfinite(self.pbest_values[min_idx]):
            self.gbest_position = self.pbest_positions[min_idx].copy()
            self.gbest_value = self.pbest_values[min_idx]
        else:
            # Fallback if no finite pbest values found initially
            self.gbest_position = self.positions[0].copy() if self.num_particles > 0 else np.zeros(self.number_dimensions)
            self.gbest_value = np.inf

    def step(self, action: np.ndarray) -> {}:
        self.previous_positions = self.positions.copy()

        # perturb
        r1 = np.random.rand(self.num_particles, self.number_dimensions)
        r2 = np.random.rand(self.num_particles, self.number_dimensions)

        # velocity components
        cognitive_velocity = c1 * r1 * (self.pbest_positions - self.positions)
        social_velocity = c2 * r2 * (self.gbest_position - self.positions)
        inertia_velocity = omega * self.velocities

        # new velocity
        self.velocities = inertia_velocity + cognitive_velocity + social_velocity

        # determine new (unbounded) positions
        self.positions += self.velocities

        # Personal Best Update
        is_out_of_bounds = np.any((self.positions < self.lower_bound) | (self.positions > self.upper_bound), axis=1)
        feasible_mask = ~is_out_of_bounds
        fitness_values = np.full(self.num_particles, np.inf, dtype=float)

        feasible_positions = self.positions[feasible_mask]
        if feasible_positions.shape[0] > 0:
            evaluated_feasible_fitness = self.objective_function.evaluate_matrix(feasible_positions)
            fitness_values[feasible_mask] = evaluated_feasible_fitness

        improvement_mask = fitness_values < self.pbest_values
        self.pbest_positions[improvement_mask] = self.positions[improvement_mask]
        self.pbest_values[improvement_mask] = fitness_values[improvement_mask]

        # global best update
        if not np.all(np.isinf(self.pbest_values)):
            current_min_idx = np.argmin(self.pbest_values)
            current_gbest_value = self.pbest_values[current_min_idx]  # This will be finite if not all are inf

            # Update gbest only if the new best pbest is better than the current gbest
            if current_gbest_value < self.gbest_value:
                self.gbest_value = current_gbest_value
                self.gbest_position = self.pbest_positions[current_min_idx].copy()
                
        # record g_best
        self._gbest_history.append(self.gbest_value)

        metrics = self._compute_metrics()
        metrics['gbest_value'] = self.gbest_value

        return metrics


    def _compute_metrics(self) -> dict:
        metrics = {
                'avg_step_size': np.nan,
                'avg_current_velocity_magnitude': np.nan,
                'swarm_diversity': np.nan,
                'infeasible_ratio': np.nan,
                'stability_ratio': np.nan,
                'gbest_value': np.nan,
            }

        # step size
        step_sizes = np.linalg.norm(self.positions - self.previous_positions, axis=1)
        metrics['avg_step_size'] = np.mean(step_sizes)

        # average velocity
        velocity_magnitudes = np.linalg.norm(self.velocities, axis=1)
        metrics['avg_current_velocity_magnitude'] = np.mean(velocity_magnitudes)

        # TODO check poli stability

        # diversity of swarm
        if self.num_particles > 1:
            centroid = np.mean(self.positions, axis=0)
            distances = np.linalg.norm(self.positions - centroid, axis=1)
            metrics['swarm_diversity'] = np.mean(distances)
        elif self.num_particles == 1:
            metrics['swarm_diversity'] = 0.0

        # out of bounds particles
        is_out_of_bounds = np.any((self.positions < self.lower_bound) | (self.positions > self.upper_bound), axis=1)
        infeasible_count = np.sum(is_out_of_bounds)
        metrics['infeasible_ratio'] = infeasible_count / self.num_particles

        return metrics

