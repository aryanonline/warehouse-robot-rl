## 🚀 Project Objective

This project explores how **Reinforcement Learning (RL)** can be applied to train a robot to efficiently navigate a **warehouse environment**. The goal is to develop an agent that:
- 🟢 **Delivers boxes to the target location** while navigating the grid.
- 🛡️ Avoids **collisions with workers and equipment**.
- ⚡ Optimizes its movements to **minimize energy consumption and maximize delivery efficiency**.
- 🎯 Adapts to stochastic uncertainties in movement (e.g., unintended shifts).

The robot operates in a dynamic grid environment, where:
1. **Workers (red squares)** move randomly, simulating real-world unpredictability.
2. **Obstacles (dark gray rectangles)** are fixed equipment in the warehouse.
3. **The robot (green circle)** must efficiently deliver boxes to the target location (purple square) and then immediately pick up another box for the next delivery.

---

## 📈 Key Findings

Through experimentation, we trained **Q-Learning** and **SARSA** agents and evaluated their performance based on:
1. **Boxes Delivered**: The number of successful deliveries.
2. **Collision Rate**: The number of collisions with workers or obstacles.
3. **Path Efficiency**: Steps taken to deliver each box.
4. **Longest Safe Duration**: The longest time the robot avoided any collisions.
5. **Cumulative Rewards**: Total reward accumulated by the agent.

### **Highlights**
- The **Q-Learning agent** demonstrated higher delivery efficiency but incurred slightly more collisions due to its exploration-heavy approach.
- The **SARSA agent** exhibited safer behavior, prioritizing stability over maximum efficiency, making it suitable for high-safety environments.

---

## 🗺 Warehouse Grid Environment

Below is a visualization of the warehouse grid environment:

- 🟢 **Green Circle**: Robot
- 🔴 **Red Squares**: Workers (randomly moving)
- ⚫ **Dark Gray Rectangle**: Equipment/Obstacles
- 🟣 **Purple Square**: Target Delivery Location

<img width="263" alt="image" src="https://github.com/user-attachments/assets/acb2b6c7-8780-417d-bdb0-4825f1aa45f1" />

---

## 🔍 Summary of Results

| Metric                  | Q-Learning Agent  | SARSA Agent         |
|-------------------------|-------------------|---------------------|
| **Boxes Delivered**     | High (Efficient)  | Medium (Safe)       |
| **Collisions**          | Moderate          | Low                 |
| **Path Efficiency**     | Optimal           | Near-Optimal        |
| **Longest Safe Duration** | Medium           | High                |
| **Cumulative Rewards**  | High              | Medium              |

---

## 🚀 Conclusion
This project demonstrates how **Reinforcement Learning** can effectively solve real-world navigation problems in dynamic environments. The comparison of **Q-Learning** and **SARSA** agents highlights the trade-offs between **efficiency** and **safety**, providing insights into deploying autonomous systems in warehouses.

Future improvements could include:
- Multi-agent training for collaborative robot teams.
- Dynamic obstacles and environmental changes.
- Incorporating advanced RL algorithms, such as Deep Q-Networks (DQN).

