import gymnasium as gym
import os
from stable_baselines3 import SAC, DDPG
from stable_baselines3.common.callbacks import CheckpointCallback

env_id = "InvertedDoublePendulum-v5"
log_dir = "./training_logs/"
model_dir = "./models/"

os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Create the environment
env = gym.make(env_id)

# Initialize Models
print("Initializing SAC model...")
sac_model = SAC(
    "MlpPolicy", 
    env, 
    verbose=1, 
    tensorboard_log=log_dir,
    learning_rate=3e-4,
    buffer_size=1000000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
)

print("Initializing DDPG model...")
ddpg_model = DDPG(
    "MlpPolicy", 
    env, 
    verbose=1, 
    tensorboard_log=log_dir,
    learning_rate=1e-3,
    buffer_size=1000000,
    batch_size=128,
    tau=0.005,
    gamma=0.99,
)

# Training Parameters
total_timesteps = 100000

# Train and Save SAC Model
print(f"\n--- Training SAC for {total_timesteps} steps ---")
sac_model.learn(total_timesteps=total_timesteps, tb_log_name="SAC_run")
sac_model.save(f"{model_dir}sac_pendulum")
print("SAC training complete and saved.")

# Train and Save DDPG Model
print(f"\n--- Training DDPG for {total_timesteps} steps ---")
ddpg_model.learn(total_timesteps=total_timesteps, tb_log_name="DDPG_run")
ddpg_model.save(f"{model_dir}ddpg_pendulum")
print("DDPG training complete and saved.")

env.close()