import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvFeatureExtractor(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),   # (1,84,84) -> (32,20,20)
            nn.BatchNorm2d(32),
            nn.Mish(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # (32,20,20) -> (64,9,9)
            nn.BatchNorm2d(64),
            nn.Mish(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1), # (64,9,9) -> (128,7,7)
            nn.BatchNorm2d(128),
            nn.Mish(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1), # (128,7,7) -> (128,5,5)
            nn.BatchNorm2d(128),
            nn.Mish(),
            nn.Flatten()
        )
        self.output_dim = 128 * 5 * 5  # Output dimension after convolutional layers

    def forward(self, x):
        return self.conv(x)

class DiscreteActor(nn.Module):
    def __init__(self, action_dim, in_channels=1):
        super().__init__()
        self.feature = ConvFeatureExtractor(in_channels=in_channels)
        self.fc = nn.Sequential(
            nn.Linear(self.feature.output_dim, 1024),
            nn.LayerNorm(1024),
            nn.Mish(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.Mish(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.Mish(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        logits = self.fc(self.feature(x))
        probs = torch.softmax(logits, dim=-1)
        return probs

    def sample(self, x):
        probs = self.forward(x)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, probs

class DiscreteCritic(nn.Module):
    def __init__(self, action_dim, in_channels=1):
        super().__init__()
        self.feature = ConvFeatureExtractor(in_channels=in_channels)
        self.fc = nn.Sequential(
            nn.Linear(self.feature.output_dim, 1024),
            nn.LayerNorm(1024),
            nn.Mish(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.Mish(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.Mish(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        value = self.fc(self.feature(x))
        return value

class SACDiscreteAgent:
    def __init__(self, state_shape, action_dim, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.8):
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        print(f"Using device: {device}")

        self.actor = DiscreteActor(action_dim, in_channels=self.state_shape[0]).to(device)
        self.critic1 = DiscreteCritic(action_dim, in_channels=self.state_shape[0]).to(device)
        self.critic2 = DiscreteCritic(action_dim, in_channels=self.state_shape[0]).to(device)
        self.target_critic1 = DiscreteCritic(action_dim, in_channels=self.state_shape[0]).to(device)
        self.target_critic2 = DiscreteCritic(action_dim, in_channels=self.state_shape[0]).to(device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.AdamW(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.AdamW(self.critic2.parameters(), lr=lr)

    def select_action(self, state, eval_mode=False):
        state = torch.FloatTensor(state).unsqueeze(0).float().to(device)  # (1, 1, 84, 84)
        probs = self.actor(state)
        if eval_mode:
            action = torch.argmax(probs, dim=-1)
            return action.item()
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            return action.item()

    def update(self, replay_buffer, batch_size=64):
        states, actions, _, rewards, next_states, dones = replay_buffer.sample(batch_size)
        # Convert to torch tensors and move to device
        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).long().unsqueeze(1).to(device)
        rewards = torch.from_numpy(rewards).float().unsqueeze(1).to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        dones = torch.from_numpy(dones).float().unsqueeze(1).to(device)

        # Critic update
        with torch.no_grad():
            next_probs = self.actor(next_states)
            next_log_probs = torch.log(next_probs + 1e-8)
            target_q1 = self.target_critic1(next_states)
            target_q2 = self.target_critic2(next_states)
            target_min_q = torch.min(target_q1, target_q2)
            next_v = (next_probs * (target_min_q - self.alpha * next_log_probs)).sum(dim=1, keepdim=True)
            target_q = rewards + (1 - dones) * self.gamma * next_v

        current_q1 = self.critic1(states).gather(1, actions)
        current_q2 = self.critic2(states).gather(1, actions)
        critic1_loss = nn.MSELoss()(current_q1, target_q)
        critic2_loss = nn.MSELoss()(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=1.0)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1.0)
        self.critic2_optimizer.step()

        # Actor update
        probs = self.actor(states)
        log_probs = torch.log(probs + 1e-8)
        q1 = self.critic1(states)
        q2 = self.critic2(states)
        min_q = torch.min(q1, q2)
        actor_loss = (probs * (self.alpha * log_probs - min_q)).sum(dim=1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # Soft update target networks
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'actor_loss': actor_loss.item()
        }