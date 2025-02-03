# 🚀 Reinforcement Learning for Warehouse Robot Navigation

## 📌 Project Overview
In modern warehouses, **autonomous robots** play a crucial role in efficiently transporting goods. This project explores how **Reinforcement Learning (RL)** can be used to train a **robot to navigate a warehouse** while **avoiding obstacles, workers, and minimizing collisions**. The goal is to **deliver as many boxes as possible** while ensuring safety and optimal energy use.

## Warehouse Grid Overview

Below is the visualization of the warehouse grid environment:

- 🟢 **Green Circle**: Robot
- 🔴 **Red Squares**: Workers
- ⚫ **Dark Gray Rectangle**: Equipment/Obstacles
- 🟣 **Purple Square**: Target Delivery Location

<img width="263" alt="image" src="https://github.com/user-attachments/assets/acb2b6c7-8780-417d-bdb0-4825f1aa45f1" />

This project simulates a **grid-based warehouse environment**, where the robot must:
- Navigate **without colliding with workers or fixed equipment**.
- Optimize **path efficiency and delivery rate**.
- Handle **stochastic movement errors** to simulate real-world uncertainty.

## ⚡️ Getting Started

### **1️⃣ Installation**
To set up the project, install the required dependencies:

```bash
# Clone the repository
git clone https://github.com/your-username/warehouse-robot-rl.git
cd warehouse-robot-rl

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
