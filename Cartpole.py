import gym
import time
import matplotlib 
import random
import numpy as np


env = gym.make("CartPole-v1", render_mode="human")
print("Onservational Space:",env.observation_space)
print("Action Space:",env.action_space)

print("Observation Space High:",env.observation_space.high)
print("Observation Space Low:",env.observation_space.low)




num_episodes = 5  # Run for 5 episodes

for episode in range(num_episodes):
    state = env.reset()
    action = env.action_space.sample()  # Random action
    env.step(action)
    done = False
    score = 0  

    while not done:
        env.render()  # ✅ This should now work correctly
        action = env.action_space.sample()  # Random action (0 or 1)
        next_state, reward, done, _, _ = env.step(action)  # Apply action
        score += reward
        time.sleep(0.02)  # Slow down for visualization
    
    print(f"Episode {episode+1} Score: {score}")

env.close()  # Close the environment after running
