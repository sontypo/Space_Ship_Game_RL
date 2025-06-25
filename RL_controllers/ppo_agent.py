import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class PolicyNetwork(nn.Module):
    def __init__(self, action_dim, in_channels=1):
        super(PolicyNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),  # (1,84,84) -> (32,20,20)
            nn.Mish(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # (32,20,20) -> (64,9,9)
            nn.Mish(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # (64,9,9) -> (64,7,7)
            nn.Mish(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.Mish(),
            nn.Linear(512, action_dim)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return torch.softmax(x, dim=-1)

class ValueNetwork(nn.Module):
    def __init__(self, in_channels=1):
        super(ValueNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.Mish(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.Mish(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.Mish(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.Mish(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x.squeeze(-1)

class PPOAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.0005, gamma=0.99, epsilon=0.2, epochs=10, batch_size=64, device=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        self.batch_size = batch_size

        self.policy_net = PolicyNetwork(action_dim=action_dim, in_channels=state_dim[0]).to(self.device)
        self.value_net = ValueNetwork(in_channels=state_dim[0]).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.learning_rate)

    def select_action(self, state):
        # state: (1, 84, 84) numpy array
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # (1, 1, 84, 84)
        probs = self.policy_net(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def compute_advantages(self, rewards, values, next_values, dones):
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_values[i] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.epsilon * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        return torch.tensor(advantages, dtype=torch.float32).to(self.device)

    def update(self, replay_buffer, batch_size=64):
        states, actions, log_probs_old, rewards, next_states, dones = replay_buffer.sample(batch_size)
        # Convert numpy arrays to tensors
        states = torch.from_numpy(states).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        # Reshape to (batch_size, channels, height, width)
        states = states.view(batch_size, *self.state_dim)
        next_states = next_states.view(batch_size, *self.state_dim)
        actions = torch.from_numpy(actions).long().unsqueeze(1).to(self.device)
        log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32).to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        dones = torch.from_numpy(dones).float().unsqueeze(1).to(self.device)

        values = self.value_net(states)
        next_values = self.value_net(next_states).detach()
        advantages = self.compute_advantages(rewards, values, next_values, dones)
        returns = advantages + values

        for _ in range(self.epochs):
            # Policy update
            probs = self.policy_net(states)
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(actions)
            ratio = torch.exp(log_probs - log_probs_old)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            # Value update
            value_loss = nn.MSELoss()(self.value_net(states), returns.detach())

            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()