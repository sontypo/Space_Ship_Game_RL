import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class DQN(nn.Module):
    def __init__(self, state_shape, action_dim):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(state_shape[0], 32, kernel_size=8, stride=4),  # (C, H, W) -> (32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # (32, 20, 20) -> (64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # (64, 9, 9) -> (64, 7, 7)
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)
    
    
class DuelingDQN(nn.Module):
    def __init__(self, state_shape, action_dim):
        super(DuelingDQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(state_shape[0], 32, kernel_size=8, stride=4),  # (C, H, W) -> (32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # (32, 20, 20) -> (64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 9, 9) -> (64, 7, 7)
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1), # (128,7,7) -> (128,5,5)
            nn.Mish(),
            nn.Flatten()
        )
        self.output_dim = 128 * 5 * 5  # Output dimension after convolutional layers
        self.fc_value = nn.Sequential(
            nn.Linear(self.output_dim, 1024),
            nn.Mish(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.Mish(),
            nn.Linear(512, 1)  # Value stream
        )
        self.fc_advantage = nn.Sequential(
            nn.Linear(self.output_dim, 1024),
            nn.Mish(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.Mish(),
            nn.Linear(512, action_dim)  # Advantage stream
        )

    def forward(self, x):
        x = self.conv(x)
        value = self.fc_value(x)
        advantage = self.fc_advantage(x)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
    

class DQNAgent:
    def __init__(self, state_shape, action_dim, replay_buffer, model_path,
                lr=1e-4, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, 
                epsilon_decay=10000, net_type='d3qn', use_prioritized_replay=False):
        
        self.net_type = net_type
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.replay_buffer = replay_buffer
        self.model_path = model_path
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.use_prioritized_replay = use_prioritized_replay
        if net_type == 'd3qn':
            print("Using Dueling Double DQN architecture")
            self.model = DuelingDQN(state_shape, action_dim).to(device)
            self.target_model = DuelingDQN(state_shape, action_dim).to(device)
        else:
            self.model = DQN(state_shape, action_dim).to(device)
            self.target_model = DQN(state_shape, action_dim).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.steps = 0
        

    def select_action(self, state, eval_mode=False):    #TODO: Checked
        if eval_mode or random.random() > self.epsilon:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = self.model(state)
            return torch.argmax(q_values, dim=1).item()
        else:
            return random.randint(0, self.action_dim - 1)
        

    def store_experience(self, state, action, reward, next_state, done):    #TODO: Checked
        # Compute initial priority (e.g., max priority so far, or 1.0 if empty)
        if self.use_prioritized_replay:
            if len(self.replay_buffer.priorities) > 0:
                priority = max(self.replay_buffer.priorities)
            else:
                priority = 1.0
            experience = (state, action, reward, next_state, done)
            self.replay_buffer.push(experience, priority)
        
        else:
            self.replay_buffer.push(state, action, 0, reward, next_state, done)
        

    def update(self, batch_size=64):    #TODO: NOT Checked
        # Check if using prioritized replay
        if self.use_prioritized_replay:
            experiences, indices = self.replay_buffer.sample(batch_size)
            # Unpack experiences
            states, actions, rewards, next_states, dones = zip(*experiences)
            states = torch.from_numpy(np.stack(states)).float().to(device)
            actions = torch.from_numpy(np.array(actions)).long().unsqueeze(1).to(device)
            rewards = torch.from_numpy(np.array(rewards)).float().unsqueeze(1).to(device)
            next_states = torch.from_numpy(np.stack(next_states)).float().to(device)
            dones = torch.from_numpy(np.array(dones).astype(np.float32)).unsqueeze(1).to(device)
            

            if not self.net_type == 'd3qn':
                q_values = self.model(states).gather(1, actions)
                next_q_values = self.target_model(next_states).max(1)[0].detach().unsqueeze(1)
                
            else:
                # Current Q values
                q_values = self.model(states).gather(1, actions)
                # Double DQN: action selection is from main model, value is from target model
                next_actions = self.model(next_states).argmax(1, keepdim=True)  # (batch_size, 1)
                next_q_values = self.target_model(next_states).gather(1, next_actions)
                
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

            # Compute TD-errors for prioritized replay
            td_errors = (q_values - target_q_values).detach().cpu().numpy().squeeze()
            new_priorities = np.abs(td_errors) + 1e-6  # small constant to avoid zero priority

            # Update priorities in memory
            self.replay_buffer.update_priorities(indices, new_priorities)
            
        else:
            states, actions, _, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
            # Convert to torch tensors and move to device
            states = torch.from_numpy(states).float().to(device)
            actions = torch.from_numpy(actions).long().unsqueeze(1).to(device)
            rewards = torch.from_numpy(rewards).float().unsqueeze(1).to(device)
            next_states = torch.from_numpy(next_states).float().to(device)
            dones = torch.from_numpy(dones).float().unsqueeze(1).to(device)
            
            if not self.net_type == 'd3qn':
                q_values = self.model(states).gather(1, actions)
                next_q_values = self.target_model(next_states).max(1)[0].detach().unsqueeze(1)
                
            else:
                # Current Q values
                q_values = self.model(states).gather(1, actions)
                # Double DQN: action selection is from main model, value is from target model
                next_actions = self.model(next_states).argmax(1, keepdim=True)  # (batch_size, 1)
                next_q_values = self.target_model(next_states).gather(1, next_actions)
                
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        self.epsilon = max(self.epsilon_end, self.epsilon_start - (self.steps / self.epsilon_decay) * (self.epsilon_start - self.epsilon_end))

        # Save loss to CSV
        loss_log_path = os.path.join(self.model_path, "loss_log.csv")
        os.makedirs(self.model_path, exist_ok=True)
        with open(loss_log_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([self.steps, loss.item()])