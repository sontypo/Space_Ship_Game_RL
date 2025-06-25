import sys
import os
# Add the parent directory of RL_controllers and space_ship_game_RL to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(BASE_DIR, "space_ship_game_RL"))
print("Base directory added to sys.path:", BASE_DIR)
import cv2
import pygame
import numpy as np
from collections import deque
from game import Game
from setting import WIDTH, HEIGHT, FPS, WHITE, GREEN  # Import constants

class SpaceShipEnv():
    def __init__(self):
        pygame.init()
        pygame.font.init()

        self.screen = None
        self.clock = pygame.time.Clock()
        self.fps = FPS

        self.game = Game()
        self.action_space = [0, 1, 2, 3]  # Define actions: 0 - No action, 1 - Move left, 2 - Move right, 3 - Shoot
        self.observation_space = (WIDTH, HEIGHT, 3)  # Assuming RGB images as observations
        self.prev_health = self.game.player.sprite.health  # Track previous health
        
        # self.frame_skip = frame_skip
        # self.stack_frames = stack_frames
        # self.frames = deque(maxlen=stack_frames)
        
    
    def reset(self):
        self.game = Game()
        return self.game.state.transpose([1, 0, 2])
    

    def step(self, action):
        self.game.update(action)
        self.game.draw()
        state = self.game.state.transpose([1, 0, 2])  # Convert to (C, H, W) format for CNN input
        reward = self.calculate_reward()
        done = not self.game.running or self.game.score >= 10000
        info = {}
        # print(f"Score: {self.game.score}, Health: {self.game.player.sprite.health}")
        score = self.game.score
        health = self.game.player.sprite.health
        # if not self.game.running:
        #     print("Game Over! Final Score:", score)
        self.prev_health = health  # Update previous health for next step

        return state, reward, done, info, score, health
    

    def render(self):
        if self.screen is None:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("SpaceShip PPO Environment")

        self.game.draw(self.screen)
        pygame.display.update()
        self.clock.tick(self.fps)
    
    
    def close(self):
        pygame.quit()
        self.game = None
        

    def calculate_reward(self):
        # Reward for score increase
        survival_reward = 0.25
        # Positive reward for collecting powerups
        powerup_reward = 0
        if self.game.last_powerup == 'shield':
            powerup_reward += 50  # You can adjust this value
        elif self.game.last_powerup == 'gun':
            powerup_reward += 25
            
        reward = survival_reward + powerup_reward
        # Penalty if got hit (health decreased)
        current_health = self.game.player.sprite.health
        if current_health < self.prev_health:
            reward -= 100  # Adjust this penalty value as needed

        return reward