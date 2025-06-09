from copy import deepcopy

import torch
from torch import nn, optim

from RL.Actor.Actor import Actor
from RL.Critic.Critic import QNetwork


class Agent:
    def __init__(
        self,
        state_dim,
        action_dim, # This should be 3 if fixed nt, 4 if adaptive
        hidden_dim=256,
        gamma=0.99, # Discount factor (Note: Paper uses gamma=1.0 for PSO env)
        tau=0.005,  # Target network soft update coefficient (Polyak averaging)
        alpha=0.2,  # Entropy regularization coefficient (temperature parameter)
        actor_lr=3e-4, # Learning rate for the actor network
        critic_lr=3e-4,# Learning rate for the critic networks
        device="cpu", # Device to run computations on ('cpu' or 'cuda')
        adaptive_nt=False # Flag to indicate if 'nt' is part of the action space
    ):
        self.device = device
        self.discount_factor = gamma
        self.slow_update_co_eff = tau
        self.entropy_weight = alpha
        self.adaptive_nt = adaptive_nt  # Store flag
        self.action_dim = action_dim  # Store correct action dim (3 or 4)

        # action selector
        self.actor = Actor(state_dim, self.action_dim, hidden_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        # dual critic network
        self.q1 = QNetwork(state_dim, self.action_dim, hidden_dim).to(device)
        self.q2 = QNetwork(state_dim, self.action_dim, hidden_dim).to(device)

        # updates are pushed to the target
        self.q1_target = deepcopy(self.q1).eval().to(device)
        self.q2_target = deepcopy(self.q2).eval().to(device)

        # optimizer step using Adam
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=critic_lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=critic_lr)

        # loss function
        self.loss_fn = nn.MSELoss()


    def sample_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if deterministic:
                # sample mean action
                action = self.actor.get_deterministic_action(state)
            else:
                # sample action from dist
                action, _ = self.actor(state)
        return action.detach().cpu().numpy()[0]

    def train_step(self, replay_buffer, batch_size):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # guess next action
            next_action, next_log_prob = self.actor(next_state)

            target_q1 = self.q1_target(next_state, next_action)
            target_q2 = self.q2_target(next_state, next_action)

            target_q_min = torch.min(target_q1, target_q2)

            target_q = target_q_min - self.entropy_weight * next_log_prob

            target = reward + (1.0 - done) * self.discount_factor * target_q

        current_q1 = self.q1(state, action)
        current_q2 = self.q2(state, action)

        q1_loss = self.loss_fn(current_q1, target)
        q2_loss = self.loss_fn(current_q2, target)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        for param in self.q1.parameters(): param.requires_grad = False
        for param in self.q2.parameters(): param.requires_grad = False

        new_action, log_prob = self.actor(state)

        q1_val = self.q1(state, new_action)
        q2_val = self.q2(state, new_action)

        q_val_min = torch.min(q1_val, q2_val)

        actor_loss = (self.entropy_weight * log_prob - q_val_min).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param in self.q1.parameters(): param.requires_grad = True
        for param in self.q2.parameters(): param.requires_grad = True

        self.slow_update_critic(self.q1, self.q1_target)
        self.slow_update_critic(self.q2, self.q2_target)

    def slow_update_critic(self, source_net, target_net):
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(self.slow_update_co_eff * source_param.data + (1.0 - self.slow_update_co_eff) * target_param.data)
