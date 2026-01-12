# This is the training code for the turbulent pendulum-v1 
import gymnasium as gym
import os
from stable_baselines3 import SAC, DDPG
from stable_baselines3.common.callbacks import BaseCallback
from wrappers import TurbulentGravityWrapper

# --- Callback to log gravity to TensorBoard ---
class GravityLoggerCallback(BaseCallback):
    def _on_step(self) -> bool:
        # Pull the gravity value from the info dict (first environment in vector)
        g_val = self.locals["infos"][0].get("current_gravity")
        if g_val is not None:
            self.logger.record("env/gravity", g_val)
        return True

# --- Configuration ---
ENV_ID = "Pendulum-v1"
LOG_DIR = "./chaos_logs/"
os.makedirs(LOG_DIR, exist_ok=True)

base_env = gym.make(ENV_ID)
env = TurbulentGravityWrapper(base_env, low=5.0, high=15.0, turbulence_std=0.2)

# Setup Models
sac_model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR)
ddpg_model = DDPG("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR)

# Train with Gravity Logging
print("Starting SAC Training...")
sac_model.learn(total_timesteps=50000, tb_log_name="Chaos_SAC_Pendulum", callback=GravityLoggerCallback())
sac_model.save("models/chaos_sac_pendulum_final")

print("Starting DDPG Training...")
ddpg_model.learn(total_timesteps=50000, tb_log_name="Chaos_DDPG_Pendulum", callback=GravityLoggerCallback())
ddpg_model.save("models/chaos_ddpg_pendulum_final")
env.close()