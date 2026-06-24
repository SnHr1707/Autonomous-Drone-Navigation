# Autonomous Vision-Based Drone Navigation via RDDPG

![Deep Learning](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Unreal Engine](https://img.shields.io/badge/unreal_engine-313131?style=for-the-badge&logo=unrealengine&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Reinforcement Learning](https://img.shields.io/badge/Reinforcement_Learning-000000?style=for-the-badge&logo=openai&logoColor=white)

An end-to-end Deep Reinforcement Learning project that trains a quadcopter to autonomously navigate a complex 3D environment, avoid obstacles, and perform precise, low-velocity target landings using only raw depth-camera vision and kinematic sensors. 

This project utilizes **Project Colosseum (AirSim fork)** and **Unreal Engine**, driven by a custom **Recurrent Deep Deterministic Policy Gradient (RDDPG)** algorithm with **Prioritized Experience Replay (PER)**.

## ✨ Key Features
* **Multi-Modal AI:** Fuses spatial features from a 4-layer CNN (processing 128x72 depth images) with a 6D kinematic state vector.
* **Temporal Memory:** Utilizes a Gated Recurrent Unit (GRU) across a 5-frame sequence to allow the agent to infer momentum, optical flow, and approach speed.
* **Dynamic Reward Shaping:** Implements a continuous PD-style penalty gradient that eliminates "dive-bombing", forcing the drone to brake smoothly to `< 1.0 m/s` upon reaching the target.
* **Prioritized Experience Replay:** Utilizes a custom SumTree data structure to sample high TD-Error experiences in $O(\log n)$ time, drastically speeding up convergence.

---

## 🛠️ Prerequisites

* **OS:** Windows 10/11 (Recommended for Unreal Engine compatibility) or Ubuntu 20.04+.
* **Hardware:** A dedicated GPU (NVIDIA RTX 2060 or better recommended) and 16GB+ RAM.
* **Software:** Python 3.8 - 3.10.

---

## 🚀 Installation & Setup Guide

### Step 1: Install Unreal Engine & AirSim (Project Colosseum)
1. Download and install the **Epic Games Launcher**.
2. Install **Unreal Engine 4.27** (or UE 5.x depending on your specific Colosseum environment).
3. Set up **Project Colosseum** (the active, maintained fork of Microsoft AirSim). Follow the official build instructions here: [Project Colosseum GitHub](https://github.com/CodexLabsLLC/Colosseum).
4. Download or create a custom Unreal Engine 3D Environment (e.g., the Blocks environment or your custom map).

### Step 2: Configure AirSim Settings
AirSim requires a specific `settings.json` file to spawn a drone instead of a car and to enable the API.
1. Navigate to your Documents folder: `C:\Users\YOUR_NAME\Documents\AirSim\`.
2. Create or edit `settings.json` to include the following configuration:
```json
{
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "ClockSpeed": 1.0,
  "Vehicles": {
    "Drone1": {
      "VehicleType": "SimpleFlight",
      "AutoCreate": true
    }
  }
}
```

### Step 3: Python Environment Setup
1. Clone this repository:
  ```bash
git clone https://github.com/SnHr1707/Autonomous-Drone-Navigation
cd Autonomous-Drone-Navigation
```
2. Create a virtual environment (optional but recommended):
  ```bash
python -m venv venv
venv\Scripts\activate
```
3. Install the required dependencies:
```bash
pip install torch torchvision torchaudio numpy opencv-python airsim gymnasium tensorboard matplotlib
```

### Step 4: 🎮 How to Run the Project
1. Start the Simulation:
Before running any Python scripts, you must start the Unreal Engine environment.
    - Open your Unreal Engine project.
    - Click the Play button in the Unreal Editor.
    - Note: The drone should spawn, and the AirSim API will begin listening for Python commands.
2. Train the Model from Scratch:
- To begin training a new RDDPG agent:
```bash
python train.py
```
- Models will be saved in the models/RDDPG/ directory every 50 episodes.
- You can monitor training metrics (Actor Loss, Critic Loss, Episode Reward) using TensorBoard:

```Bash
tensorboard --logdir=logs/RDDPG
```
3. Resume Training (Auto-Recovery):
- If the simulation crashes or you wish to resume training from your highest episode:
```Bash
python train_mid.py
```
- This script automatically detects the latest checkpoint (or SUCCESS checkpoint) and resumes seamlessly. It also contains the Early Stopping logic that halts training once the drone successfully brakes at the target 3 times in a row.
4. Test the Trained AI
- Once trained, you can watch the AI fly autonomously without any random exploration noise:
```bash
python test.py
```
- Ensure you update test.py with the exact filename of your best model (e.g., actor_SUCCESS_ep2528.pth).
5. Generate 3D Flight Trajectory Video: To visualize the AI's learning progression over time, run the video generator. This script loads all saved models, plots their flight paths in 3D space, and creates an animated .mp4 comparing early random flights (grey) to the final optimized flight path (red).
```Bash
python make_video.py
```

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| `colosseum_env.py` | Custom Gymnasium environment bridging AirSim API, physics state, and reward shaping. |
| `rddpg.py` | PyTorch implementation of the Actor-Critic networks (CNN + GRU) and the RDDPG agent logic. |
| `per.py` | Implementation of the SumTree data structure and Prioritized Experience Replay memory buffer. |
| `train.py` | Main execution loop for training the agent from scratch. |
| `train_mid.py` | Advanced training loop with auto-resume, streak-tracking, and early-stopping functionality. |
| `test.py` | Evaluation script for deploying the trained model with live terminal telemetry. |
| `make_video.py` | Matplotlib script to generate an animated 3D comparison of flight trajectories. |
