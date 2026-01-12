# This is the training code for pendulum-v1 using SAC

import gymnasium as gym
from stable_baselines3 import SAC
import os

tb_log_dir = "./sac_tb_logs/"
os.makedirs(tb_log_dir, exist_ok=True)

env = gym.make("Pendulum-v1", render_mode="rgb_array")

model = SAC(
    policy="MlpPolicy", 
    env=env, 
    ent_coef='auto',
    verbose=1, 
    device="auto",
    tensorboard_log=tb_log_dir 
)
# Train
model.learn(total_timesteps=50000, tb_log_name="SAC_pendulum_run")

model.save("sac_pendulum_model")
print("Training finished and logged!")