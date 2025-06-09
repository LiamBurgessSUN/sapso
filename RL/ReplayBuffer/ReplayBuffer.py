import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim, device="cpu"):
        self.buffer_capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.state_dim = state_dim  # Store state dimension
        self.action_dim = action_dim  # Store action dimension
        self.device = device

    def store_episode(self, state, action, reward, next_state, done):
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        action = np.array(action, dtype=np.float32)
        reward = float(reward)
        done = float(done)
        self.buffer.append((state, action, reward, next_state, done))

    def sample_batch(self, batch_size):
        sampled_batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*sampled_batch))
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.float32).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
        done = torch.tensor(done, dtype=torch.float32).unsqueeze(1).to(self.device)

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
