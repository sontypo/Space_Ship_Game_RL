import numpy as np
import argparse
import torch
import torch.optim as optim
from dqn_agent import DQNAgent
from env_wrapper import SpaceShipEnv
from utils import stack_frames, log_metrics, save_model, create_directory, ReplayBuffer

# usage: python3 train_dqn_normal_buffer.py --max_episodes 4000 --save_path d3qn_normal_buffer_data_2
def train_dqn():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_episodes', type=int, default='4000', help="Set maximum number of episodes for training.")
    parser.add_argument('--save_path', type=str, default='d3qn_normal_buffer_data_2', help='Loading save path.')
    args = parser.parse_args()
    max_episodes = args.max_episodes
    model_dir = 'saved_models/' + args.save_path
    create_directory(model_dir)
    
    env = SpaceShipEnv()
    action_dim = len(env.action_space)
    state_shape = (4, 84, 84)  # Assuming grayscale images of size 84x84
    learning_rate = 1e-4
    use_prioritized_replay = False  # Set to False to use normal Replay Buffer
    replay_buffer =  ReplayBuffer(capacity=50000)
    agent = DQNAgent(state_shape=state_shape, action_dim=action_dim, replay_buffer=replay_buffer, model_path=model_dir, lr=learning_rate, net_type='d3qn', use_prioritized_replay=use_prioritized_replay)

    max_episodes = 4000
    batch_size = 64
    
    reward_history = []
    score_history = []
    total_healths = []
    timestep = 0
    best_score = -float('inf')  # Add this before the training loop
    best_episode = 0
    print("Training DQN Agent (Normal Buffer) ...")
    for episode in range(max_episodes):
        state = env.reset()
        state, stacked_frames = stack_frames(None, state, True)
        done = False
        total_reward = 0
        current_score = 0
        current_health = 0

        while not done:
            action = agent.select_action(state, eval_mode=False)
            next_state, reward, done, _, current_score, current_health = env.step(action)
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

            agent.store_experience(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            timestep += 1

            if agent.replay_buffer.__len__() >= batch_size:
                agent.update(batch_size)

        reward_history.append(total_reward)
        score_history.append(current_score)
        total_healths.append(current_health)
        if current_score > best_score:
            best_score = current_score
            best_episode = episode
            
            torch.save({
                'model': agent.model.state_dict(),
                'target_model': agent.target_model.state_dict(),
                'optimizer': agent.optimizer.state_dict(),
                'episode': episode,
                'total_steps': timestep,
                'best_score': best_score,
                'reward_history': reward_history,
                'score_history': score_history,
            }, 'checkpoint.pth')
                
        if timestep % 1000 == 0:
            agent.target_model.load_state_dict(agent.model.state_dict())

        print(f"Episode: {episode+1}, Total Reward: {total_reward}, Score: {current_score}, Health: {current_health}, Memory Size: {agent.replay_buffer.__len__()}")
        # Save data to directory
        np.savetxt(model_dir + '/reward_per_episode.csv', reward_history, delimiter = ' , ')
        np.savetxt(model_dir + '/score_per_episode.csv', score_history, delimiter = ' , ')
        np.savetxt(model_dir + '/health_per_episode.csv', total_healths, delimiter = ' , ')
        # Log metrics
        log_metrics(metrics={'episode': episode+1, 'reward': total_reward, 'score': current_score}, folder=model_dir)

        if (episode + 1) % 50 == 0:
            save_model(agent.model, f"{model_dir}/d3qn_model_ep{episode+1}.pth")
            save_model(agent.target_model, f"{model_dir}/d3qn_target_model_ep{episode+1}.pth")

    env.close()
    print(f"Training completed!!! At episode: {best_episode + 1}, highest score achieved: {best_score}.")

if __name__ == "__main__":
    train_dqn()
