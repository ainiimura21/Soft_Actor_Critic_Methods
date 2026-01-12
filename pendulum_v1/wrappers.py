# This is the wrapper code for the turbulent pendulum-v1 
import gymnasium as gym
import numpy as np

class TurbulentGravityWrapper(gym.Wrapper):
    def __init__(self, env, low=5.0, high=15.0, turbulence_std=0.5):
        super().__init__(env)
        self.low = low
        self.high = high
        self.turbulence_std = turbulence_std

    def reset(self, seed=None, options=None):
        # Set a random baseline gravity
        new_gravity = np.random.uniform(self.low, self.high)
        self.env.unwrapped.g = new_gravity
        
        obs, info = self.env.reset(seed=seed, options=options)
        # Pass gravity into info for the logger
        info["current_gravity"] = new_gravity
        return obs, info

    def step(self, action):
        # Add turbulence noise to gravity
        change = np.random.normal(0, self.turbulence_std)
        current_g = self.env.unwrapped.g
        
        new_g = np.clip(current_g + change, self.low, self.high)
        self.env.unwrapped.g = new_g 
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Add the new gravity to the info dict for logging
        info["current_gravity"] = new_g
        return obs, reward, terminated, truncated, info