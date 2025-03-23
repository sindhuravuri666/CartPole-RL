import gym
import time


env = gym.make("CartPole-v1", render_mode="human")

num_episodes = 5  # Run for 5 episodes

for episode in range(num_episodes):
    state = env.reset()
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
