import gym
import numpy as np
import random
import time

# Create CartPole environment
env = gym.make("CartPole-v1", render_mode= "human")

def discretize_state(state, bins):
    """Discretizes continuous state space into bins."""
    cart_position_bins = np.linspace(-2.4, 2.4, bins[0] - 1)
    cart_velocity_bins = np.linspace(-2, 2, bins[1] - 1)
    pole_angle_bins = np.linspace(-0.5, 0.5, bins[2] - 1)
    pole_velocity_bins = np.linspace(-3.5, 3.5, bins[3] - 1)
    
    cart_position_idx = np.digitize(state[0], cart_position_bins) - 1
    cart_velocity_idx = np.digitize(state[1], cart_velocity_bins) - 1
    pole_angle_idx = np.digitize(state[2], pole_angle_bins) - 1
    pole_velocity_idx = np.digitize(state[3], pole_velocity_bins) - 1

    return (cart_position_idx, cart_velocity_idx, pole_angle_idx, pole_velocity_idx)

# Define Q-learning parameters
num_states = (10, 10, 10, 10)  # Discretized state space
num_actions = env.action_space.n  # 2 actions (left or right)
q_table = np.zeros(num_states + (num_actions,))  # Q-table

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Initial exploration rate
epsilon_decay = 0.995  # Decay rate per episode
min_epsilon = 0.01  # Minimum exploration rate
episodes = 5000  # Number of training episodes
bins = (10, 10, 10, 10)  # Define discretization bins

# Training loop
episode_rewards = []  # Track rewards per episode
q_value_updates = []  # Track Q-value updates

for episode in range(episodes):
    state = discretize_state(env.reset()[0], bins)  # Reset environment and get initial state
    done = False
    total_reward = 0

    while not done:
        # Choose action: Exploration vs Exploitation
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Random action (explore)
        else:
            action = np.argmax(q_table[state])  # Best action (exploit)
        
        # Take action, observe next state and reward
        next_state, reward, done, _, _ = env.step(action)
        next_state = discretize_state(next_state, bins)
        total_reward += reward

        # Q-learning update rule
        best_next_action = np.argmax(q_table[next_state])
        old_q_value = q_table[state + (action,)]
        new_q_value = old_q_value + alpha * (
            reward + gamma * q_table[next_state + (best_next_action,)] - old_q_value
        )

        # Store Q-value update
        q_value_updates.append(abs(new_q_value - old_q_value))

        # Update Q-table
        q_table[state + (action,)] = new_q_value
        state = next_state  # Move to next state

    episode_rewards.append(total_reward)  # Track total reward
    epsilon = max(min_epsilon, epsilon * epsilon_decay)  # Decay epsilon

    # Print progress every 500 episodes
    if (episode + 1) % 500 == 0:
        avg_q_update = np.mean(q_value_updates[-500:]) if len(q_value_updates) >= 500 else np.mean(q_value_updates)
        avg_reward = np.mean(episode_rewards[-500:]) if len(episode_rewards) >= 500 else np.mean(episode_rewards)
        print(f"Episode {episode + 1} | Avg Reward: {avg_reward:.2f} | Epsilon: {epsilon:.4f} | Avg Q-Update: {avg_q_update:.5f}")

# Testing the trained agent
print("\nTesting the trained agent...\n")
state = discretize_state(env.reset()[0], bins) 
done = False
total_reward = 0

while not done:
    action = np.argmax(q_table[state])  # Use learned policy
    next_state, reward, done, _, _ = env.step(action)
    next_state = discretize_state(next_state, bins)
    state = next_state
    total_reward += reward
    env.render()  # Render the environment

print(f"Final Score: {total_reward}")
env.close()
