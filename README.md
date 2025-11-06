# Multi-Agent Reinforcement Learning with Hybrid Control

This repository implements a multi-agent reinforcement learning (MARL) framework for training agents with a hybrid control approach using Graph Neural Networks (GNNs) and mode-based strategic supervision. The agents learn to navigate an environment while balancing speed and safety using three discrete control modes: CRUISE, RUSH, and AVOID.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Key Components](#key-components)
- [Training](#training)
- [Results](#results)
- [License](#license)

## Overview
This project demonstrates a MARL system where multiple agents coordinate actions based on their own states and communication graphs represented using PyTorch Geometric. The agents operate under distinct modes that influence their behavior:
- **CRUISE:** Balanced speed and safety (default)
- **RUSH:** Prioritize speed when time is critical
- **AVOID:** Prioritize safety when collision risk is high

The training uses Soft Actor-Critic (SAC) style centralized training with decentralized execution, leveraging replay buffers, GNN-based actor networks, and a centralized critic.

## Features
- Multi-agent environment with configurable number of agents and world size
- Communication graph construction based on agent proximity
- Mode-based strategic control supervision (CRUISE, RUSH, AVOID)
- Graph Neural Networks for encoding agent interactions
- Soft Actor-Critic (SAC) based multi-agent training
- Replay buffer for experience storage and sampling
- Training progress visualization with success, reward, and collision curves
- Device-agnostic (CPU/GPU) training support

## Installation

1. Clone this repository:
```
git clone https://github.com/Zuru07/MARL-Hybrid-Control.git
cd marl-hybrid-control
```

2. Create and activate a Python environment (recommended):

```
python3 -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

> Note: Installing PyTorch Geometric may require specific versions depending on your CUDA setup. Refer to the [official installation guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

## Usage

Run the main training script with:

```
python CTDE_MA_SAC.py
```


This will start training 4 agents for 3000 episodes on the default environment and display progress logs and plots.

## Code Structure

- **`MultiAgentEnvironment`**: Simulates the multi-agent world including agent positions, goals, step dynamics, reward calculations, and communication graphs.
- **`AgentMode`**: Enumeration of discrete agent control modes used for strategic supervision.
- **`ReplayBuffer`**: Experience replay buffer for storing transitions (including modes).
- **`GNNLayer` & `GraphEncoder`**: Graph neural network layers encoding agent interactions.
- **`Actor`**: Actor network per agent with state, mode embeddings, and GNN-based graph context.
- **`Critic`**: Centralized critic network estimating joint Q-values.
- **`MultiAgentTrainer`**: Orchestrates training including mode supervision, action selection, centralized updates, and plotting.

## Key Components

### Environment
- Agents move in a bounded 2D world with random initial and goal positions.
- They have velocity limits and discrete action spaces normalized between -1 and 1.
- Communications are defined via a radius; edges correspond to agents close enough to communicate.

### Modes
- `CRUISE`: Normal navigation.
- `RUSH`: Activated during last 25% of allowed time, prioritizing faster movement.
- `AVOID`: Triggered when agents are dangerously close, prioritizing safety.

### Neural Networks
- **Actor:** Inputs include agent state, graph embeddings, and mode embeddings; outputs action distribution parameters.
- **Critic:** Joint state-action inputs for centralized Q-value estimation.

### Training Loop
- Agents interact with environment, supervisor assigns modes.
- Transitions (including modes) stored in replay buffer.
- Batch updates optimize actor, critic, and temperature parameters with SAC loss.
- Logs and plots track reward, success, and collision metrics.

## Training

- Train for 3000 episodes by default with 4 agents.
- Logs print every 50 episodes showing reward averages, success rates, and collisions.
- Visualization saved as `training_curves_hybrid.png`.

## Results

After training, the system outputs final average metrics:
- Success rate over last 100 episodes: 98%
- Average episode reward over last 100 episodes: 600

Training curves show trends of performance improvement and collision reduction over time and this can be viewed on the **training_curves.png**.

## License
MIT License — see [LICENSE](LICENSE) file for details.

---

For questions or contributions, feel free to open an issue or submit a pull request.