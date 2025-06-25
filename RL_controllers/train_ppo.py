import numpy as np
from env_wrapper import SpaceShipEnv
from ppo_agent import PPOAgent, torch
from utils import stack_frames, log_metrics, save_model, create_directory, ReplayBuffer

def train_ppo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    st_f = 4
    learning_rate = 5e-4
    env = SpaceShipEnv()
    # state_dim = np.array(state).flatten().shape[0]
    action_dim = len(env.action_space)
    agent = PPOAgent(state_dim=(st_f, 84, 84), action_dim=action_dim, learning_rate=learning_rate, device=device)
    buffer = ReplayBuffer(capacity=100000)

    max_episodes = 4000
    update_timestep = 2048
    model_dir = "saved_models/ppo_data"
    batch_size = 128
    create_directory(model_dir)
    
    reward_history = []
    score_history = []
    total_healths = []

    timestep = 0
    best_score = -float('inf')  # Add this before the training loop
    best_episode = 0
    print("Training PPO Agent ...")
    for episode in range(max_episodes):
        state = env.reset()
        state, stacked_frames = stack_frames(None, state, True)
        done = False
        total_reward = 0
        current_score = 0
        current_health = 0

        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, done, _, current_score, current_health = env.step(action)
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            
            buffer.push(
                np.array(state).flatten(),
                action,
                log_prob.item(),  # Store log_prob for PPO
                reward,
                np.array(next_state).flatten(),
                float(done)
            )

            state = next_state
            total_reward += reward
            timestep += 1
            
            # Update PPO agent
            if len(buffer) >= batch_size and timestep % update_timestep == 0:
                agent.update(buffer, batch_size)
        
        reward_history.append(total_reward)
        score_history.append(current_score)
        total_healths.append(current_health)
        if current_score > best_score:
            best_score = current_score
            best_episode = episode
            
            if (episode + 1) % 50 == 0:
                torch.save({
                    'policy_net': agent.policy_net.state_dict(),
                    'value_net': agent.value_net.state_dict(),
                    'policy_optimizer': agent.policy_optimizer.state_dict(),
                    'value_optimizer': agent.value_optimizer.state_dict(),
                    'episode': episode,
                    'total_steps': timestep,
                    'best_score': best_score,
                    'reward_history': reward_history,
                    'score_history': score_history,
                }, 'checkpoint.pth')

        print(f"Episode: {episode+1}, Total Reward: {total_reward}, Score: {current_score}, Health: {current_health}, Memory Size: {buffer.__len__()}")
        # Save data to directory
        np.savetxt(model_dir + '/reward_per_episode.csv', reward_history, delimiter = ' , ')
        np.savetxt(model_dir + '/score_per_episode.csv', score_history, delimiter = ' , ')
        np.savetxt(model_dir + '/health_per_episode.csv', total_healths, delimiter = ' , ')
        # Log metrics
        log_metrics(metrics={'episode': episode+1, 'reward': total_reward, 'score': current_score}, folder=model_dir)

        if (episode + 1) % 50 == 0:
            save_model(agent.policy_net, f"{model_dir}/sac_policy_net_ep{episode+1}.pth")
            save_model(agent.value_net, f"{model_dir}/sac_value_net_ep{episode+1}.pth")

    env.close()
    print(f"Training completed!!! At episode: {best_episode + 1}, highest score achieved: {best_score}.")

if __name__ == "__main__":
    train_ppo()