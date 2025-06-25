import torch
import imageio
from dqn_agent import DQNAgent, device
from utils import np, pygame, stack_frames, PrioritizedReplayMemory, ReplayBuffer, create_directory
from env_wrapper import SpaceShipEnv

def validate_env_specific_checkpoint(env, using_prioritized_replay, checkpoint=None):
    if using_prioritized_replay:
        replay_buffer =  PrioritizedReplayMemory(capacity=50000)
        model_path = 'saved_models/d3qn_data_2/d3qn_model_ep'  + checkpoint + '.pth'   #
        target_model_path = 'saved_models/d3qn_data_2/d3qn_target_model_ep' + checkpoint + '.pth'
    else:
        replay_buffer = ReplayBuffer(capacity=50000)
        model_path = 'saved_models/d3qn_normal_buffer_data_2/d3qn_model_ep'  + checkpoint +'.pth'  #
        target_model_path = 'saved_models/d3qn_normal_buffer_data_2/d3qn_target_model_ep' + checkpoint + '.pth'  #3050 > 3250
        
    action_dim = len(env.action_space)
    state_shape = (4, 84, 84)

    # Initialize agent (make sure net_type and other args match your training)
    agent = DQNAgent(state_shape=state_shape, action_dim=action_dim, replay_buffer=replay_buffer, model_path=model_path, net_type='d3qn', use_prioritized_replay=True)

    # Load checkpoint
    model_checkpoint = torch.load(model_path, map_location=device)
    target_model_checkpoint = torch.load(target_model_path, map_location=device)
    agent.model.load_state_dict(model_checkpoint)
    agent.target_model.load_state_dict(target_model_checkpoint)

    # Set agent to evaluation mode
    agent.model.eval()
    results = []
    results = []

    state = env.reset()
    state, stacked_frames = stack_frames(None, state, True)
    done = False
    total_reward = 0
    score = 0
    frames = []
    env.render()  # Ensure window is created

    while not done:
        action = agent.select_action(state, eval_mode=True)
        next_state, reward, done, _, score, health = env.step(action)
        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
        state = next_state
        total_reward += reward
        env.render()  # Optional: comment out if running headless
        
        # Capture frame for video
        surface = pygame.display.get_surface()
        frame = pygame.surfarray.array3d(surface)
        frame = np.transpose(frame, (1, 0, 2))
        frames.append(frame)

    print(f"Total reward: {total_reward}, Score: {score}")
    results.append((total_reward, score))

    env.close()
    
    # Save best trial video
    return frames, score
        
if __name__ == "__main__":
    # Create environment
    env = SpaceShipEnv()
    
    # Validate with specific checkpoint
    checkpoint = '3450'  # Change this to the desired checkpoint number
    frames, score = validate_env_specific_checkpoint(env, using_prioritized_replay=True, checkpoint=checkpoint)