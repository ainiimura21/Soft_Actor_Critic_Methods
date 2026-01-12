# This is the testing code for the turbulent pendulum-v1 
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC, DDPG
from stable_baselines3.common.logger import configure
from wrappers import TurbulentGravityWrapper 
import time
import pandas as pd
import matplotlib.pyplot as plt

# Setup Environment
env_id = "Pendulum-v1"
base_env = gym.make(env_id, render_mode="human")
env = TurbulentGravityWrapper(base_env, low=5.0, high=15.0, turbulence_std=2)

# Setup TensorBoard Logger for the test phase
log_path = "./test_logs_pendulum/"
test_logger = configure(log_path, ["stdout", "tensorboard"])

# Load Models
try:
    sac_model = SAC.load("models/chaos_sac_pendulum_final")
    ddpg_model = DDPG.load("models/chaos_ddpg_pendulum_final")
except FileNotFoundError:
    print("Error: Model files not found. Ensure they are in the './models/' folder.")
    exit()

def run_test(model, name, episodes=20):
    print(f"\n--- Testing {name} ---")
    model.set_logger(test_logger)
    episode_rewards = []
    
    for ep in range(episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        step_count = 0
        
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Log gravity to TensorBoard live
            model.logger.record(f"test_{name}/gravity", info["current_gravity"])
            model.logger.record(f"test_{name}/step_reward", reward)
            
            time.sleep(0.01)
            step_count += 1
            
            # Dump logs every 20 steps to avoid slowing down too much
            if step_count % 20 == 0:
                model.logger.dump(step=step_count + (ep * 200))
            
        episode_rewards.append(total_reward)
        print(f"Episode {ep+1} Total Reward: {total_reward:.2f}")
    
    return episode_rewards

# Results
results = {}
results["SAC"] = run_test(sac_model, "SAC")
results["DDPG"] = run_test(ddpg_model, "DDPG")

env.close()

df = pd.DataFrame(results)
df.to_csv("pendulum_v1_results.csv", index_label="Episode")

# Create comparison Boxplot
plt.figure(figsize=(10, 6))
plt.boxplot([results["SAC"], results["DDPG"]], labels=["SAC", "DDPG"])
plt.title(f"Performance on {env_id} with Turbulent Gravity")
plt.ylabel("Cumulative Reward")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig("pendulum_v1_comparison.png")
plt.show()

print("\n--- Final Statistics ---")
summary = df.describe().loc[['mean', 'std', 'min']]
print(summary)
print("\nTesting complete. View live logs with: tensorboard --logdir ./test_logs_pendulum/")