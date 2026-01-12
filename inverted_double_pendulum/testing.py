import gymnasium as gym
from stable_baselines3 import SAC, DDPG
import time

# Setup the environment with rendering enabled
env_id = "InvertedDoublePendulum-v5"
env = gym.make(env_id, render_mode="human")

# Load the trained models
model_path_sac = "./models/sac_pendulum.zip"
model_path_ddpg = "./models/ddpg_pendulum.zip"

print("Loading models...")
sac_agent = SAC.load(model_path_sac)
ddpg_agent = DDPG.load(model_path_ddpg)

def test_agent(model, name, num_episodes=3):
    print(f"\n--- Testing {name} Agent ---")
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0
        
        while not (done or truncated):
            # Predict the action (deterministic=True is best for testing)
            action, _states = model.predict(obs, deterministic=True)
            
            # Apply action to environment
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            # Slow down slightly so we can watch it
            env.render()
            time.sleep(0.01) 
            
        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")

#  Run the tests
test_agent(sac_agent, "SAC")
test_agent(ddpg_agent, "DDPG")

env.close()