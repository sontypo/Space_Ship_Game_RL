import numpy as np
from env_wrapper import SpaceShipEnv
from SAC_discrete_agent import SACDiscreteAgent
from utils import stack_frames, log_metrics, save_model, create_directory, ReplayBuffer

def train_sac_discrete():
    st_f = 4
    env = SpaceShipEnv()
    action_dim = len(env.action_space)
    learning_rate = 5e-4
    agent = SACDiscreteAgent(state_shape=(st_f, 84, 84), action_dim=action_dim, lr=learning_rate)
    buffer = ReplayBuffer(capacity=100000)

    max_episodes = 5000
    update_timestep = 2048
    batch_size = 128
    model_dir = "saved_models/sac_discrete"
    create_directory(model_dir)
    
    total_rewards = []
    total_scores = []
    total_healths = []

    timestep = 0
    best_score = -float('inf')
    best_episode = 0
    print("Training SAC-Discrete Agent ...")

    for episode in range(max_episodes):
        state = env.reset()
        # state = preprocess_image(state)
        state, stacked_frames = stack_frames(None, state, True)
        done = False
        total_reward = 0
        current_score = 0
        current_health = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, current_score, current_health = env.step(action)
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

            buffer.push(
                state,  # already shape (1,84,84)
                action,
                0,  # log_prob not needed for SAC
                reward,
                next_state,
                float(done)
            )

            state = next_state
            total_reward += reward
            timestep += 1

            # Update SAC agent
            if len(buffer) >= batch_size and timestep % update_timestep == 0:
                agent.update(buffer, batch_size)

        if current_score > best_score:
            best_score = current_score
            best_episode = episode

        total_rewards.append(total_reward)
        total_scores.append(current_score)
        total_healths.append(current_health)
        print(f"Episode: {episode+1}, Total Reward: {total_reward}, Score: {current_score}, Health: {current_health}, Memory Size: {buffer.__len__()}")
        # Save data to directory
        np.savetxt(model_dir + '/reward_per_episode.csv', total_rewards, delimiter = ' , ')
        np.savetxt(model_dir + '/score_per_episode.csv', total_scores, delimiter = ' , ')
        np.savetxt(model_dir + '/health_per_episode.csv', total_healths, delimiter = ' , ')
        # Log metrics
        log_metrics(metrics={'episode': episode+1, 'reward': total_reward, 'score': current_score}, folder=model_dir)

        if (episode + 1) % 100 == 0:
            save_model(agent.actor, f"{model_dir}/sac_discrete_actor_ep{episode+1}.pth")
            save_model(agent.critic1, f"{model_dir}/sac_discrete_critic1_ep{episode+1}.pth")
            save_model(agent.critic2, f"{model_dir}/sac_discrete_critic2_ep{episode+1}.pth")

    env.close()
    print(f"Training completed!!! At episode: {best_episode + 1}, highest score achieved: {best_score}.")

if __name__ == "__main__":
    train_sac_discrete()