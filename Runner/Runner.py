from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

from PSO.Environment.ObjectiveFunctions.Training.Loader import objective_function_classes
from PSO.Agent.Agent import Agent as PsoAgent
from RL.Agent.Agent import Agent as RLAgent
from RL.ReplayBuffer.ReplayBuffer import ReplayBuffer


class ConfigurationParameters:
    def __init__(self):
        self.episodes_per_function = 30
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.buffer_capacity = 1_000_000
        self.environment_dimensions = 3
        self.start_steps=1000
        self.batch_size = 256
        self.start_steps = 1000
        self.updates_per_step = 1
        self.graphics_path = Path(__file__).parent / "Graphics"


class Runner:

    def __init__(self):
        self.pso_environment = PsoAgent(
            num_particles=30,
            max_steps=5000,
            agent_step_size=10,
            adaptive_nt=False
        )

        self.config = ConfigurationParameters()

        state_dim = self.pso_environment.observation_space.shape[0]
        action_dim = self.pso_environment.action_space.shape[0]

        self.buffer = ReplayBuffer(
            capacity=self.config.buffer_capacity,
            state_dim=state_dim,
            action_dim=action_dim,
            device=self.config.device
        )

        self.rl_agent = RLAgent(
            state_dim=state_dim,
            action_dim=action_dim,
        )

    def run(self):
        results_log = {}  # {func_name: [(ep1_reward, ep1_gbest), ...]}
        global_step_count = 0
        total_agent_steps = 0
        total_episodes_run = 0

        # Loop through environments
        for i, objective_function_class in enumerate(objective_function_classes):

            # initialize environment
            self.pso_environment.load_environment(
                objective_function_class(dimensions=self.config.environment_dimensions, ))

            # initialize results
            results_log[objective_function_class.__name__] = []

            # number of episodes / attempts within an environments
            for episode_num in range(self.config.episodes_per_function):
                episode_reward = 0.0
                terminated, truncated = False, False
                episode_agent_steps = 0

                # reset environment
                state, _ = self.pso_environment.reset()

                episode_best_gbest = self.pso_environment.gbest_value

                # episode interaction
                while not terminated and not truncated:
                    if total_agent_steps < self.config.start_steps:
                        action = self.pso_environment.action_space.sample()
                    else:
                        state_np = np.array(state, dtype=np.float32)

                        # sample action from RL agent
                        action = self.rl_agent.sample_action(state_np)

                    next_state, reward, terminated, truncated, info = self.pso_environment.step(action)

                    turn_final_gbest = info.get('final_gbest', np.inf)

                    if np.isfinite(turn_final_gbest):
                        episode_best_gbest = min(episode_best_gbest, turn_final_gbest)

                    episode_agent_steps += 1
                    total_agent_steps += 1
                    pso_steps_this_turn = info.get('steps_taken', 0)
                    global_step_count += pso_steps_this_turn

                    # episode info for buffer
                    done_flag = terminated or truncated

                    self.buffer.store_episode(np.array(state, dtype=np.float32),
                                np.array(action, dtype=np.float32),
                                float(reward),
                                np.array(next_state, dtype=np.float32),
                                bool(done_flag))

                    state = next_state
                    episode_reward += reward

                    # train RL agent
                    if len(self.buffer) >= self.config.batch_size and total_agent_steps >= self.config.start_steps:
                        for _ in range(self.config.updates_per_step):
                            self.rl_agent.train_step(self.buffer, self.config.batch_size)

                if not np.isfinite(episode_best_gbest):
                    final_gbest_for_episode = self.pso_environment.gbest_value
                else:
                    final_gbest_for_episode = episode_best_gbest

                results_log[objective_function_class.__name__].append((episode_reward, final_gbest_for_episode))

                self.pso_environment.close()

        # compute graphics for functions

        # compute average reward per function
        avg_rewards_per_func = {}
        for func_name, results in results_log.items():
            avg_rewards_per_func[func_name] = np.mean(replace_infinite_rewards(results))

            plt.figure(figsize=(12, 6))

            valid_func_names = [name for name, avg in avg_rewards_per_func.items() if np.isfinite(avg)]
            valid_avg_rewards = [avg for avg in avg_rewards_per_func.values() if np.isfinite(avg)]

            if valid_func_names:
                plt.bar(valid_func_names, valid_avg_rewards)
                plt.title(f"SAC Avg Reward per Function ({self.config.episodes_per_function} eps each)")
                plt.xlabel("Objective Function")
                plt.ylabel(f"Average Reward over Episodes")
                plt.xticks(rotation=90)
                plt.grid(axis='y')
                plt.tight_layout()
                plt.savefig(str(
                    self.config.graphics_path / f"{func_name}_average_train_rewards.png"
                ))
                plt.close()

def replace_infinite_rewards(results):
    rewards = [r for r, g in results]
    new_rewards = list(rewards)
    for i, r in enumerate(rewards):
        if not np.isfinite(r):
            prev_r = None
            if i > 0:
                for j in range(i - 1, -1, -1):
                    if np.isfinite(rewards[j]):
                        prev_r = rewards[j]
                        break
            next_r = None
            if i < len(rewards) - 1:
                for j in range(i + 1, len(rewards)):
                    if np.isfinite(rewards[j]):
                        next_r = rewards[j]
                        break
            if prev_r is not None and next_r is not None:
                new_rewards[i] = np.median([prev_r, next_r])
            else:
                new_rewards[i] = 0.0
    return new_rewards

