import numpy as np
import pygame
import imageio
import cv2
from PIL import Image
import os
import torch
import random
from collections import deque

def preprocess_image(image):
    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Resize to 84x84
    image = cv2.resize(image, (84, 84))
    image = image / 255.0
    return np.expand_dims(image, axis=0)

def preprocess_frame(frame):
    image = Image.fromarray(frame)
    image = image.convert('L')
    # Use correct resampling attribute for your Pillow version
    try:
        resample = Image.Resampling.BILINEAR
    except AttributeError:
        resample = Image.BILINEAR
    image = image.resize((84, 84), resample)
    frame = np.asarray(image, dtype=np.float32) / 255.0

    return frame

def stack_frames(stacked_frames, state, is_new_episode):
    frame = preprocess_frame(state)
    if is_new_episode or stacked_frames is None:
        # If it's a new episode or no previous frames, initialize with 4 identical frames
        stacked_frames = deque([frame]*4, maxlen=4)
    else:
        stacked_frames.append(frame)
    # Stack the 4 frames along the first dimension: shape becomes (4, 84, 84)
    stacked_state = np.stack(stacked_frames, axis=0)

    return stacked_state, stacked_frames

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0

    def push(self, state, action, log_prob, reward, next_state, done):
        transition = (state, action, log_prob, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            # Replace the oldest transition
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        states, actions, log_probs, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, log_probs, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()

def save_model(model, path):
    """
    Save the model's state dictionary to the specified path.
    """
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """
    Load the model's state dictionary from the specified path.
    """
    model.load_state_dict(torch.load(path))
    model.eval()

def log_metrics(metrics, folder, filename='training_log.txt'):
    """
    Log training metrics to a file.
    """
    create_directory(folder)
    with open(os.path.join(folder, filename), 'a') as f:
        f.write(f"{metrics}\n")

def create_directory(path):
    """
    Create a directory if it does not exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def record_episode_as_video(env, done, video_path="gameplay.mp4"):
    frames = []
    while not done:
        surface = pygame.display.get_surface()
        frame = pygame.surfarray.array3d(surface)  # shape: (W, H, 3)
        frame = np.transpose(frame, (1, 0, 2))     # pygame 是 x,y → imageio 是 y,x
        frames.append(frame)

    imageio.mimsave(video_path, frames, fps=60, quality=9)
    print(f"Saved gameplay video to: {video_path}")
    

class PrioritizedReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.priorities = []
        self.position = 0

    def push(self, experience, priority):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
            self.priorities.append(priority)
        else:
            self.memory[self.position] = experience
            self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        if len(self.memory) == 0:
            return [], []

        priorities = np.array(self.priorities, dtype=np.float32)
        probabilities = priorities / priorities.sum()
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        experiences = [self.memory[i] for i in indices]
        return experiences, indices

    def update_priorities(self, indices, priorities):
        for index, priority in zip(indices, priorities):
            self.priorities[index] = priority