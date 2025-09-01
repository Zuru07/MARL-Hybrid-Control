import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import random
from collections import deque, namedtuple
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# --- Core Components ---

# --- CHANGE START ---
# Transition now stores joint states and actions for the shared replay buffer
Transition = namedtuple('Transition', ['joint_state', 'joint_action', 'reward', 'joint_next_state', 'done'])
# --- CHANGE END ---

class ReplayBuffer:
    """A shared replay buffer for all agents."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, *args):
        self.buffer.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# --- Agent Networks (Actor & Centralized Critic) ---

# Graph Neural Network for Actor communication
class GNNLayer(MessagePassing):
    """GNN layer for message passing between agents."""
    def __init__(self, input_dim, output_dim):
        super(GNNLayer, self).__init__(aggr='mean')
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, output_dim), nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)
    
    def message(self, x_i, x_j):
        return self.mlp(torch.cat([x_i, x_j], dim=-1))

class GraphEncoder(nn.Module):
    """Encodes node features using multiple GNN layers."""
    def __init__(self, node_dim, hidden_dim, output_dim):
        super(GraphEncoder, self).__init__()
        self.gnn1 = GNNLayer(node_dim, hidden_dim)
        self.gnn2 = GNNLayer(hidden_dim, output_dim)
        
    def forward(self, x, edge_index):
        x = F.relu(self.gnn1(x, edge_index))
        x = self.gnn2(x, edge_index)
        return x

class Actor(nn.Module):
    """Individual agent's policy network, uses GNN for context."""
    def __init__(self, state_dim, action_dim, hidden_dim=256, gnn_dim=64):
        super(Actor, self).__init__()
        self.gnn_dim = gnn_dim
        self.graph_encoder = GraphEncoder(state_dim, hidden_dim // 2, gnn_dim)
        
        self.fc1 = nn.Linear(state_dim + gnn_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
        self.max_log_std = 2
        self.min_log_std = -20
    
    def forward(self, state, graph_data=None):
        if graph_data is not None and graph_data.edge_index.numel() > 0:
            graph_embed = self.graph_encoder(graph_data.x, graph_data.edge_index)
            agent_graph_embed = graph_embed[graph_data.agent_index].unsqueeze(0)
            x = torch.cat([state, agent_graph_embed], dim=-1)
        else:
            zero_embed = torch.zeros(state.shape[0], self.gnn_dim, device=state.device)
            x = torch.cat([state, zero_embed], dim=-1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        
        return mean, log_std
    
    def sample(self, state, graph_data=None):
        mean, log_std = self.forward(state, graph_data)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

# --- CHANGE START ---
class Critic(nn.Module):
    """Centralized critic network that observes all agents."""
    def __init__(self, num_agents, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        # Input is all states and all actions concatenated
        input_dim = num_agents * (state_dim + action_dim)
        
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, all_states, all_actions):
        # Concatenate states and actions, then flatten for the MLP
        x = torch.cat([all_states, all_actions], dim=-1)
        x = x.view(x.size(0), -1)  # Flatten from [batch, num_agents, features] to [batch, num_agents * features]
        return self.q1(x), self.q2(x)
# --- CHANGE END ---

# --- Simulation Environment ---

class MultiAgentEnvironment:
    def __init__(self, num_agents=4, world_size=10.0, communication_radius=4.0, max_episode_steps=100):
        self.num_agents = num_agents
        self.world_size = world_size
        self.communication_radius = communication_radius
        self.max_episode_steps = max_episode_steps
        
        self.state_dim = 5
        self.action_dim = 2
        
        self.max_speed = 2.0
        self.dt = 0.1
        self.collision_radius = 0.5
        self.goal_radius = 0.3
        
        self.reset()
    
    def reset(self):
        self.positions = np.random.uniform(-self.world_size / 2, self.world_size / 2, (self.num_agents, 2))
        self.velocities = np.zeros((self.num_agents, 2))
        self.goals = np.random.uniform(-self.world_size / 2, self.world_size / 2, (self.num_agents, 2))
        
        distances_to_goal = np.linalg.norm(self.goals - self.positions, axis=1)
        # --- CHANGE --- : Shorter, more challenging time limits
        self.time_limits = distances_to_goal / (self.max_speed * 0.8) + 3.0
        self.time_remaining = self.time_limits.copy()
        
        self.step_count = 0
        self.done = False
        
        return self.get_observations()
    
    def get_observations(self):
        """Returns a normalized observation for each agent."""
        observations = []
        for i in range(self.num_agents):
            # --- CHANGE --- : Normalize inputs to be in a consistent range [-1, 1]
            pos_norm = self.positions[i] / (self.world_size / 2)
            vel_norm = self.velocities[i] / self.max_speed
            time_norm = np.clip(self.time_remaining[i] / self.time_limits[i], 0, 1) * 2 - 1 # to [-1, 1]

            obs = np.concatenate([pos_norm, vel_norm, [time_norm]])
            observations.append(obs)
        return np.array(observations)
    
    def get_communication_graph(self):
        edge_indices = []
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if np.linalg.norm(self.positions[i] - self.positions[j]) <= self.communication_radius:
                    edge_indices.extend([[i, j], [j, i]])
        if not edge_indices:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.LongTensor(edge_indices).t().contiguous()
    
    def step(self, actions):
        actions = np.clip(actions, -1, 1)
        self.velocities += actions * self.dt
        
        speeds = np.linalg.norm(self.velocities, axis=1)
        mask = speeds > self.max_speed
        if np.any(mask):
            self.velocities[mask] *= self.max_speed / speeds[mask, np.newaxis]
        
        self.positions += self.velocities * self.dt
        self.positions = np.clip(self.positions, -self.world_size / 2, self.world_size / 2)
        
        self.time_remaining -= self.dt
        self.step_count += 1
        
        reward = self.calculate_rewards()
        self.check_termination()
        
        return self.get_observations(), reward, self.done, {}
    
    def calculate_rewards(self):
        """Calculates a shared team reward with stronger signals."""
        team_reward = 0.0
        
        # --- CHANGE --- : Strengthened reward/penalty signals
        goal_reward_val = 50.0
        success_bonus_val = 100.0
        timeout_penalty_val = -20.0
        collision_penalty_val = -5.0
        step_penalty = -0.05

        team_reward += step_penalty

        num_collisions = 0
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if np.linalg.norm(self.positions[i] - self.positions[j]) < self.collision_radius:
                    num_collisions += 1
        team_reward += collision_penalty_val * num_collisions

        all_agents_succeeded = True
        for i in range(self.num_agents):
            if np.linalg.norm(self.positions[i] - self.goals[i]) < self.goal_radius:
                if self.time_remaining[i] > 0:
                    team_reward += goal_reward_val # Give reward only once per agent
                    self.goals[i] = self.positions[i] # "Deactivate" goal
            else:
                all_agents_succeeded = False

            if self.time_remaining[i] <= 0:
                team_reward += timeout_penalty_val

        if all_agents_succeeded:
            team_reward += success_bonus_val
        
        return team_reward

    def check_termination(self):
        if np.any(self.time_remaining <= 0) or self.step_count >= self.max_episode_steps:
            self.done = True

    def render(self, save_path=None):
        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        ax.set_xlim(-self.world_size / 2, self.world_size / 2)
        ax.set_ylim(-self.world_size / 2, self.world_size / 2)
        
        for i in range(self.num_agents):
            color = f'C{i}'
            ax.add_patch(plt.Circle(self.positions[i], self.collision_radius / 2, color=color, alpha=0.8))
            ax.add_patch(plt.Circle(self.goals[i], self.goal_radius, color=color, fill=False, linestyle='--', alpha=0.8))
            ax.text(self.positions[i, 0], self.positions[i, 1], str(i), ha='center', va='center', color='white', fontsize=10)

        plt.title(f'Step: {self.step_count}')
        if save_path: plt.savefig(save_path, dpi=150)
        else: plt.show()
        plt.close()

# --- Training Orchestrator ---

class MultiAgentTrainer:
    """Manages the centralized training and decentralized execution process."""
    def __init__(self, env, num_episodes):
        self.env = env
        self.num_episodes = num_episodes
        self.num_agents = env.num_agents
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- CHANGE --- : Create individual actors but a single shared critic and replay buffer
        self.actors = [Actor(env.state_dim, env.action_dim).to(self.device) for _ in range(self.num_agents)]
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=3e-4) for actor in self.actors]
        
        self.critic = Critic(self.num_agents, env.state_dim, env.action_dim).to(self.device)
        self.target_critic = Critic(self.num_agents, env.state_dim, env.action_dim).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

        self.memory = ReplayBuffer(100000)
        # --- CHANGE END ---

        self.batch_size = 256
        self.tau = 0.005
        self.gamma = 0.99
        self.alpha = 0.2

        self.episode_rewards = []
        self.success_rates = []

    # def select_joint_action(self, states, graph_data, evaluate=False):
    #     """Selects a joint action for all agents, with optional hybrid supervisor."""
    #     actions = []
    #     with torch.no_grad():
    #         for i in range(self.num_agents):
    #             # --- CHANGE START --- : Hybrid Rule-Based Supervisor
    #             time_left = self.env.time_remaining[i]
    #             pos = self.env.positions[i]
                
    #             # Find nearest neighbor distance
    #             min_dist = float('inf')
    #             for j in range(self.num_agents):
    #                 if i != j:
    #                     dist = np.linalg.norm(pos - self.env.positions[j])
    #                     if dist < min_dist:
    #                         min_dist = dist
                
    #             # Rule 1: Emergency collision avoidance
    #             if min_dist < self.env.collision_radius * 1.5:
    #                 # Move away from the average position of other agents
    #                 avg_pos_others = np.mean([self.env.positions[j] for j in range(self.num_agents) if i != j], axis=0)
    #                 avoid_vec = pos - avg_pos_others
    #                 action_vec = avoid_vec / (np.linalg.norm(avoid_vec) + 1e-6)
    #             # Rule 2: Rush to goal if time is low
    #             elif time_left < 2.0: # 2 seconds threshold
    #                 goal_vec = self.env.goals[i] - pos
    #                 action_vec = goal_vec / (np.linalg.norm(goal_vec) + 1e-6)
    #             # Default: Use the learned actor policy
    #             else:
    #                 state_tensor = torch.FloatTensor(states[i]).unsqueeze(0).to(self.device)
    #                 agent_graph_data = Data(x=graph_data.x, edge_index=graph_data.edge_index, agent_index=i).to(self.device)
    #                 if evaluate:
    #                     mean, _ = self.actors[i](state_tensor, agent_graph_data)
    #                     action_tensor = torch.tanh(mean)
    #                 else:
    #                     action_tensor, _ = self.actors[i].sample(state_tensor, agent_graph_data)
    #                 action_vec = action_tensor.cpu().numpy().flatten()
                
    #             actions.append(action_vec)
    #             # --- CHANGE END ---

    #     return np.array(actions)
    # In MultiAgentTrainer class
    
    def select_joint_action(self, states, graph_data, evaluate=False):
        """Selects a joint action for all agents."""
        actions = []
        with torch.no_grad():
            for i in range(self.num_agents):
                # --- CHANGE: TEMPORARILY DISABLE SUPERVISOR ---
                # The original supervisor logic is commented out for this test.
                # We now ALWAYS use the learned actor policy.
                state_tensor = torch.FloatTensor(states[i]).unsqueeze(0).to(self.device)
                agent_graph_data = Data(x=graph_data.x, edge_index=graph_data.edge_index, agent_index=i).to(self.device)
                if evaluate:
                    mean, _ = self.actors[i](state_tensor, agent_graph_data)
                    action_tensor = torch.tanh(mean)
                else:
                    action_tensor, _ = self.actors[i].sample(state_tensor, agent_graph_data)
                action_vec = action_tensor.cpu().numpy().flatten()
                actions.append(action_vec)
        return np.array(actions)

    def train(self):
        for episode in range(self.num_episodes):
            states = self.env.reset()
            episode_reward = 0
            
            for step in range(self.env.max_episode_steps):
                edge_index = self.env.get_communication_graph().to(self.device)
                node_features = torch.FloatTensor(states).to(self.device)
                graph_data = Data(x=node_features, edge_index=edge_index)
                
                actions = self.select_joint_action(states, graph_data)
                next_states, reward, done, _ = self.env.step(actions)
                
                self.memory.push(states, actions, reward, next_states, done)
                
                states = next_states
                episode_reward += reward
                
                if len(self.memory) > self.batch_size:
                    # --- CHANGE --- : Single centralized update per step
                    self.update_centralized()

                if done: break

            # success = 1.0 if np.all([np.linalg.norm(self.env.positions[i] - self.env.goals[i]) < self.env.goal_radius for i in range(self.num_agents)]) else 0.0
            success = np.mean([np.linalg.norm(self.env.positions[i] - self.env.goals[i]) < self.env.goal_radius for i in range(self.num_agents)])
            self.episode_rewards.append(episode_reward)
            self.success_rates.append(success)

            if episode > 0 and episode % 50 == 0:
                avg_reward = np.mean(self.episode_rewards[-50:])
                avg_success = np.mean(self.success_rates[-50:])
                print(f"Episode {episode}, Avg Reward (last 50): {avg_reward:.2f}, Avg Success (last 50): {avg_success:.2f}")

    def update_centralized(self):
        """Performs a single update for the centralized critic and all actors."""
        batch = self.memory.sample(self.batch_size)
        
        # Unpack batch and convert to tensors
        joint_states = torch.FloatTensor(np.array([t.joint_state for t in batch])).to(self.device)
        joint_actions = torch.FloatTensor(np.array([t.joint_action for t in batch])).to(self.device)
        rewards = torch.FloatTensor([t.reward for t in batch]).unsqueeze(1).to(self.device)
        joint_next_states = torch.FloatTensor(np.array([t.joint_next_state for t in batch])).to(self.device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.bool).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            next_actions_list, log_probs_list = [], []
            for i in range(self.num_agents):
                # NOTE: For simplicity, GNN is not used in batch update, only for live action selection
                action, log_prob = self.actors[i].sample(joint_next_states[:, i, :], graph_data=None)
                next_actions_list.append(action)
                log_probs_list.append(log_prob)

            joint_next_actions = torch.stack(next_actions_list, dim=1)
            joint_next_log_probs = torch.cat(log_probs_list, dim=1).sum(dim=1, keepdim=True)
            
            q1_next, q2_next = self.target_critic(joint_next_states, joint_next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * joint_next_log_probs
            q_target = rewards + self.gamma * q_next * (~dones)

        q1, q2 = self.critic(joint_states, joint_actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Individual Actors
        for i in range(self.num_agents):
            new_actions, log_probs = self.actors[i].sample(joint_states[:, i, :], graph_data=None)
            
            temp_joint_actions = joint_actions.clone()
            temp_joint_actions[:, i, :] = new_actions

            q1_new, q2_new = self.critic(joint_states, temp_joint_actions)
            q_new = torch.min(q1_new, q2_new)
            
            actor_loss = (self.alpha * log_probs - q_new).mean()
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()

        # Update Target Critic Network
        with torch.no_grad():
            for target, source in zip(self.target_critic.parameters(), self.critic.parameters()):
                target.data.mul_(1.0 - self.tau)
                target.data.add_(self.tau * source.data)
                
    def plot_training_curves(self):
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Avg Reward (last 50)', color='tab:blue')
        ax1.plot(np.convolve(self.episode_rewards, np.ones(50)/50, mode='valid'), color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Avg Success Rate (last 50)', color='tab:green')
        ax2.plot(np.convolve(self.success_rates, np.ones(50)/50, mode='valid'), color='tab:green')
        ax2.tick_params(axis='y', labelcolor='tab:green')
        
        fig.tight_layout()
        plt.title('Training Progress')
        plt.savefig('training_curves.png', dpi=150)
        plt.show()

# --- Main Execution ---

if __name__ == "__main__":
    env = MultiAgentEnvironment(num_agents=4)
    trainer = MultiAgentTrainer(env, num_episodes=500)
    
    print("Starting training...")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    trainer.train()
    trainer.plot_training_curves()
    
    # Final demonstration can be added here if needed
    print("Training completed!")