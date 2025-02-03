import time
import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from robot_environment import RobotEnv
import rl_agents

# define rewards function
rewards = {"state": 0, "target": 10, "worker": -100, "equipment": -10, "wall": -5, "default": -1}

# initialize the environment with rewards and max_steps
env = RobotEnv(rewards = rewards, max_steps=1000)

# set the RL algorithm used to train the agent
rl_algo = "SARSA"

# initialize the agent
if rl_algo == "Q-Learning":
    agent = rl_agents.QLearningAgent(env, gamma=0.75, alpha=0.1, epsilon=1.0, epsilon_decay=0.99, episodes=10000)
    agent.train()
elif rl_algo == "SARSA":
    agent = rl_agents.SARSAgent(env, gamma=0.75, alpha=0.1, epsilon=1.0, epsilon_decay=0.99, episodes=10000)
    agent.train()

# TODO: Initialize variables to track performance metrics
# Metrics to include:
# 1. Number of boxes successfully delivered
# 2. Count of harmful actions or collisions
# 3. Longest uninterrupted safe duration
boxes_delivered_per_episode = []
collisions_per_episode = []
rewards_per_episode = []
longest_safe_duration_per_episode = []

num_boxes_delivered = 0
num_collisions = 0
longest_safe_duration = 0
current_safe_duration = 0

# Extra metrics for plotting
episodes = []
avg_steps_per_delivery = []
success_rates = []
collision_rates = []

# reset the environment and get the initial observation
observation, info = env.reset(seed=42), {}

# TODO: Initialize variables to track environment metrics
# Example: cumulative rewards, episode duration, etc.
#episode_rewards = 0
episode_steps = 0
cumulative_reward = 0
episode_count = 0

# run the environment until terminated or truncated
terminated, truncated = False, False

for episode in range(50):
    observation, info = env.reset(seed=42), {}
    terminated, truncated = False, False
    episode_steps = 0
    cumulative_reward = 0
    num_boxes_delivered = 0
    num_collisions = 0
    longest_safe_duration = 0
    current_safe_duration = 0

    while not terminated and not truncated:
        action = agent.choose_action(observation)  # Use trained policy
        observation, reward, terminated, truncated, info = env.step(action)

        # TODO: Update variables to calculate performance and environment metrics based on the new observation
        episode_steps += 1
        cumulative_reward += reward

        if reward == rewards["target"]:
            num_boxes_delivered += 1
            current_safe_duration += 1
        elif reward in (rewards["worker"], rewards["equipment"], rewards["wall"]):
            num_collisions += 1
            current_safe_duration = 0
        else:
            current_safe_duration += 1

        if current_safe_duration > longest_safe_duration:
            longest_safe_duration = current_safe_duration

        print(f"Step: {episode_steps}, Observation: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated} episode count: {episode + 1}")

        # render the environment at each step
        env.render()
        # Add a delay to slow down the simulation for better visualization
        time.sleep(0.1)

    # Store episode results
    episodes.append(episode + 1)
    boxes_delivered_per_episode.append(num_boxes_delivered)
    collisions_per_episode.append(num_collisions)
    rewards_per_episode.append(cumulative_reward)
    longest_safe_duration_per_episode.append(longest_safe_duration)

    avg_steps_per_delivery.append(episode_steps / (num_boxes_delivered + 1))  # Avoid division by zero
    success_rates.append((num_boxes_delivered / (episode_steps + 1)) * 100)
    collision_rates.append((num_collisions / (episode_steps + 1)) * 100)





# while not terminated and not truncated:
#     # use the agent's policy to choose an action
#     action = agent.choose_action(observation)
#     # step through the environment with the chosen action
#     observation, reward, terminated, truncated, info = env.step(action)
#
#     # TODO: Update variables to calculate performance and environment metrics based on the new observation
#     #episode_rewards += reward
#     episode_steps += 1
#     cumulative_reward += reward
#
#     if reward == rewards["target"]:
#         num_boxes_delivered += 1
#         current_safe_duration += 1
#     elif reward in (rewards["worker"], rewards["equipment"], rewards["wall"]):
#         num_collisions += 1
#         current_safe_duration = 0
#     else:
#         current_safe_duration += 1
#
#     if current_safe_duration > longest_safe_duration:
#         longest_safe_duration = current_safe_duration
#
#     # print the current state
#     print(f"Step: {episode_steps}, Observation: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated} episode count: {episode_count}")
#     # render the environment at each step
#     env.render()
#     # Add a delay to slow down the simulation for better visualization
#     time.sleep(0.1)
#
#     # reset the environment if terminated or truncated
#     if terminated or truncated:
#         print("\nTERMINATED OR TRUNCATED, RESETTING...\n")
#
#         # TODO: Update metrics for completed episode
#         #print(f"Episode {episode_count + 1}: Steps={episode_steps}, Rewards={episode_rewards}")
#         boxes_delivered_per_episode.append(num_boxes_delivered)
#         collisions_per_episode.append(num_collisions)
#         rewards_per_episode.append(cumulative_reward)
#         episode_count += 1
#         observation, info = env.reset(), {}
#
#         # TODO: Reset tracking variables for the new episode
#         #episode_rewards = 0
#         terminated, truncated = False, False
#         episode_steps = 0
#         cumulative_reward = 0
#         num_boxes_delivered = 0
#         num_collisions = 0
#         longest_safe_duration = 0
#         current_safe_duration = 0


# close the environment
env.render(close=True)

# TODO: Evaluate performance based on high-level metrics

print("\n=== PERFORMANCE EVALUATION ===")
print(f"Total Episodes: {50}")
print(f"Total Boxes Delivered: {sum(boxes_delivered_per_episode)}")
print(f"Total Collisions: {sum(collisions_per_episode)}")
print(f"Longest Safe Duration: {max(longest_safe_duration_per_episode)} steps")  # FIXED: Now correctly referenced
print(f"Average Reward per Episode: {np.mean(rewards_per_episode):.2f}")

# Boxes Delivered per Episode
plt.figure(figsize=(10, 5))
plt.plot(episodes, boxes_delivered_per_episode, label="Boxes Delivered", marker='o', color='blue')
plt.xlabel("Episode")
plt.ylabel("Boxes Delivered")
plt.title("Boxes Delivered per Episode")
plt.legend()
plt.grid()
plt.savefig("warehouse_rl_boxes_delivered-SARSA.png")
plt.show()

# Collisions per Episode
plt.figure(figsize=(10, 5))
plt.plot(episodes, collisions_per_episode, label="Collisions", marker='o', color='red')
plt.xlabel("Episode")
plt.ylabel("Collisions")
plt.title("Collisions per Episode")
plt.legend()
plt.grid()
plt.savefig("warehouse_rl_collisions-SARSA.png")
plt.show()

# Cumulative Rewards per Episode
plt.figure(figsize=(10, 5))
plt.plot(episodes, rewards_per_episode, label="Cumulative Rewards", marker='o', color='green')
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Rewards per Episode")
plt.legend()
plt.grid()
plt.savefig("warehouse_rl_rewards-SARSA.png")
plt.show()

# Longest Safe Duration per Episode
plt.figure(figsize=(10, 5))
plt.plot(episodes, longest_safe_duration_per_episode, label="Longest Safe Duration", marker='o', color='purple')
plt.xlabel("Episode")
plt.ylabel("Safe Duration (steps)")
plt.title("Longest Safe Duration per Episode")
plt.legend()
plt.grid()
plt.savefig("warehouse_rl_safe_duration-SARSA.png")
plt.show()

# Average Steps per Delivery
plt.figure(figsize=(10, 5))
plt.plot(episodes, avg_steps_per_delivery, label="Avg Steps per Delivery", marker='o', color='orange')
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.title("Avg Steps per Delivery Over Episodes")
plt.legend()
plt.grid()
plt.savefig("warehouse_rl_steps_per_delivery-SARSA.png")
plt.show()

print("Evaluation complete! Performance plots saved.")
