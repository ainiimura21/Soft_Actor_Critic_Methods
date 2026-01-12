import gymnasium as gym
import matplotlib.pyplot as plt

rewards_history = []
env = gym.make("Pendulum-v1", render_mode="human", max_episode_steps=99999)

obs, info = env.reset()


try:
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rewards_history.append(reward)
        
        if terminated or truncated:
            obs, info = env.reset()
            

except KeyboardInterrupt:
    print("\nPendulum Terminated")
finally:
    env.close()
plt.plot(rewards_history)
plt.title("Reward per Step (Random Agent)")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.show()