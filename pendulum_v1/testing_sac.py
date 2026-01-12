# This is the testing code for pendulum-v1 using SAC

import gymnasium as gym
from stable_baselines3 import SAC

env = gym.make("Pendulum-v1", render_mode="human")

# This looks for the 'sac_pendulum_model.zip' 
model = SAC.load("sac_pendulum_model")

obs, info = env.reset()
for _ in range(1000):


    action, _states = model.predict(obs, deterministic=True)
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    # If the episode ends (200 steps for Pendulum), reset and go again
    if terminated or truncated:
        obs, info = env.reset()

env.close()
print("Testing complete.")