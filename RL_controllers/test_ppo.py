class PPOAgent:
    def __init__(self, state_dim, action_dim, learning_rate=3e-4, gamma=0.99, epsilon=0.2, K_epochs=4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.K_epochs = K_epochs

        self.policy_net = self.build_network()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

    def build_network(self):
        return nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim),
            nn.Softmax(dim=-1)
        )

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probabilities = self.policy_net(state)
        action = np.random.choice(self.action_dim, p=probabilities.detach().numpy()[0])
        return action

    def update(self, states, actions, rewards, next_states, dones):
        for _ in range(self.K_epochs):
            old_probs = self.policy_net(torch.FloatTensor(states)).gather(1, torch.LongTensor(actions).unsqueeze(1)).detach()
            new_probs = self.policy_net(torch.FloatTensor(states)).gather(1, torch.LongTensor(actions).unsqueeze(1))

            ratios = new_probs / (old_probs + 1e-10)
            advantages = rewards - self.compute_value(states)

            surrogate1 = ratios * advantages
            surrogate2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
            loss = -torch.min(surrogate1, surrogate2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def compute_value(self, states):
        # Placeholder for value function computation
        return np.zeros(len(states))

def test_agent(env, agent, episodes=5):
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state

        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

if __name__ == "__main__":
    import gym
    from space_ship_game_RL.env_wrapper import SpaceShipEnv

    env = SpaceShipEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPOAgent(state_dim, action_dim)
    test_agent(env, agent)