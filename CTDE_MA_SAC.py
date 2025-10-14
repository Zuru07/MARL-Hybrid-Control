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
from enum import Enum 
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Core Components 

class AgentMode(Enum):
    CRUISE = 0  # Default behavior, balance speed and safety
    RUSH = 1    # Time is critical, prioritize speed
    AVOID = 2   # Collision is imminent, prioritize safety

# CHANGED: Added 'modes' to the Transition tuple
Transition = namedtuple('Transition', ['joint_state', 'joint_action', 'reward', 'joint_next_state', 'done', 'edge_index', 'next_edge_index', 'modes'])

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    def push(self, *args): self.buffer.append(Transition(*args))
    def sample(self, batch_size): return random.sample(self.buffer, batch_size)
    def __len__(self): return len(self.buffer)

# --- Agent Networks (Actor & Centralized Critic) ---

class GNNLayer(MessagePassing):
    def __init__(self, input_dim, output_dim):
        super(GNNLayer, self).__init__(aggr='mean')
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)
    def message(self, x_i, x_j):
        return self.mlp(torch.cat([x_i, x_j], dim=-1))

class GraphEncoder(nn.Module):
    def __init__(self, node_dim, hidden_dim, output_dim):
        super(GraphEncoder, self).__init__()
        self.gnn1 = GNNLayer(node_dim, hidden_dim)
        self.gnn2 = GNNLayer(hidden_dim, output_dim)
    def forward(self, x, edge_index):
        x = F.relu(self.gnn1(x, edge_index))
        x = self.gnn2(x, edge_index)
        return x

class Actor(nn.Module):
    # CHANGED: Added mode parameters to __init__
    def __init__(self, state_dim, action_dim, hidden_dim=256, gnn_dim=64, num_modes=3, mode_embed_dim=16):
        super(Actor, self).__init__()
        self.gnn_dim = gnn_dim
        self.graph_encoder = GraphEncoder(state_dim, hidden_dim // 2, gnn_dim)
        
        # NEW: Embedding layer for the discrete modes
        self.mode_embedding = nn.Embedding(num_modes, mode_embed_dim)
        
        # CHANGED: Input dimension for fc1 now includes the mode embedding
        self.fc1 = nn.Linear(state_dim + gnn_dim + mode_embed_dim, hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        self.max_log_std = 2
        self.min_log_std = -20
    
    # CORRECTED: The forward pass logic is now robust to batching.
    def forward(self, state, mode, graph_data=None):
        batch_size = state.shape[0]

        if graph_data is not None and graph_data.edge_index.numel() > 0:
            graph_embed = self.graph_encoder(graph_data.x, graph_data.edge_index)
            # When batch_size is 1 (during episode rollout), we must select the correct agent's embedding.
            if batch_size == 1 and graph_data.agent_index is not None:
                 # The indexing [agent_index] makes it 1D, so we .unsqueeze(0) to add the batch dim back.
                 agent_graph_embed = graph_embed[graph_data.agent_index].unsqueeze(0)
            else:
                 # This branch would be used for batched graph data, which isn't implemented here.
                 agent_graph_embed = graph_embed
        else:
            # Fallback for no graph connection or during the batch update from replay buffer.
            agent_graph_embed = torch.zeros(batch_size, self.gnn_dim, device=state.device)
        
        # Embed the mode and concatenate
        mode_embed = self.mode_embedding(mode)
        
        x = torch.cat([state, agent_graph_embed, mode_embed], dim=-1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        return mean, log_std
    
    # CHANGED: Sample method now accepts and passes the 'mode'
    def sample(self, state, mode, graph_data=None):
        mean, log_std = self.forward(state, mode, graph_data)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob
   
    
    # CHANGED: Sample method now accepts and passes the 'mode'
    def sample(self, state, mode, graph_data=None):
        mean, log_std = self.forward(state, mode, graph_data)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob

class Critic(nn.Module):
    # This class remains unchanged, as the critic is not conditioned on modes
    def __init__(self, num_agents, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
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
        x = torch.cat([all_states, all_actions], dim=-1)
        x = x.view(x.size(0), -1)
        return self.q1(x), self.q2(x)

# --- Simulation Environment ---
# The environment class remains unchanged. The supervisor logic will read from it.
class MultiAgentEnvironment:
    def __init__(self, num_agents=4, world_size=10.0, communication_radius=4.0, max_episode_steps=500):
        self.num_agents = num_agents
        self.world_size = world_size
        self.communication_radius = communication_radius
        self.max_episode_steps = max_episode_steps
        self.state_dim = 5
        self.action_dim = 2
        self.max_speed = 2.0
        self.dt = 0.1
        self.collision_radius = 0.5
        self.current_episode = 0
        self.reset()
    
    def reset(self):
        # Curriculum learning
        if self.current_episode < 500:
            self.goal_radius = 2.0
            time_buffer = 15.0
        elif self.current_episode < 1500:
            self.goal_radius = 1.0
            time_buffer = 10.0
        else:
            self.goal_radius = 0.5
            time_buffer = 5.0
        
        self.positions = self._generate_separated_positions()
        self.velocities = np.zeros((self.num_agents, 2))
        self.goals = self._generate_separated_positions()
        
        distances_to_goal = np.linalg.norm(self.goals - self.positions, axis=1)
        self.time_limits = distances_to_goal / (self.max_speed * 0.6) + time_buffer
        self.time_remaining = self.time_limits.copy()
        
        self.step_count = 0
        self.done = False
        self.prev_distances = distances_to_goal.copy()
        
        return self.get_observations()
    
    def _generate_separated_positions(self, min_separation=2.0):
        positions = []
        max_attempts = 1000
        for i in range(self.num_agents):
            for attempt in range(max_attempts):
                pos = np.random.uniform(-self.world_size / 2, self.world_size / 2, 2)
                if len(positions) == 0:
                    positions.append(pos)
                    break
                distances = [np.linalg.norm(pos - p) for p in positions]
                if min(distances) >= min_separation:
                    positions.append(pos)
                    break
            else:
                positions.append(np.random.uniform(-self.world_size / 2, self.world_size / 2, 2))
        return np.array(positions)
    
    def get_observations(self):
        observations = []
        for i in range(self.num_agents):
            rel_goal = (self.goals[i] - self.positions[i]) / self.world_size
            vel_norm = self.velocities[i] / self.max_speed
            time_norm = np.clip(self.time_remaining[i] / self.time_limits[i], 0, 1)
            obs = np.concatenate([rel_goal, vel_norm, [time_norm]])
            observations.append(obs)
        return np.array(observations)
    
    def get_communication_graph(self):
        pos_diff = self.positions[:, np.newaxis, :] - self.positions[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(pos_diff, axis=-1)
        adj_matrix = (dist_matrix > 0) & (dist_matrix <= self.communication_radius)
        edge_indices = np.array(np.where(adj_matrix)).T
        if edge_indices.shape[0] == 0:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.LongTensor(edge_indices).t().contiguous()
    
    def step(self, actions):
        actions = np.clip(actions, -1, 1)
        self.velocities = actions * self.max_speed
        self.positions += self.velocities * self.dt
        self.positions = np.clip(self.positions, -self.world_size / 2, self.world_size / 2)
        
        self.time_remaining -= self.dt
        self.step_count += 1
        
        reward = self.calculate_rewards()
        self.check_termination()
        
        return self.get_observations(), reward, self.done, {}
    
    def calculate_rewards(self):
        reward, step_cost, collision_penalty, progress_reward, success_bonus, timeout_penalty, urgency_penalty = 0.0, -0.1, -20.0, 10.0, 500.0, -100.0, -5.0
        reward += step_cost
        num_collisions = 0
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if np.linalg.norm(self.positions[i] - self.positions[j]) < self.collision_radius:
                    num_collisions += 1
        reward += collision_penalty * num_collisions
        
        current_distances = np.linalg.norm(self.positions - self.goals, axis=1)
        for i in range(self.num_agents):
            if current_distances[i] >= self.goal_radius:
                reward += progress_reward * (self.prev_distances[i] - current_distances[i])
        self.prev_distances = current_distances.copy()
        
        agents_at_goal = current_distances < self.goal_radius
        if np.all(agents_at_goal):
            reward += success_bonus
            self.done = True
        
        time_critical = self.time_remaining < (self.time_limits * 0.2)
        for i in range(self.num_agents):
            if time_critical[i] and current_distances[i] > self.goal_radius:
                reward += urgency_penalty
        
        if np.any(self.time_remaining <= 0):
            reward += timeout_penalty * (np.sum(~agents_at_goal) / self.num_agents)
        return reward

    def check_termination(self):
        if np.any(self.time_remaining <= 0) or self.step_count >= self.max_episode_steps or self.done:
            self.done = True

# --- Training Orchestrator ---

class MultiAgentTrainer:
    def __init__(self, env, num_episodes):
        self.env = env
        self.num_episodes = num_episodes
        self.num_agents = env.num_agents
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # NEW: Define number of modes for the actor
        self.num_modes = len(AgentMode)
        
        # CHANGED: Initialize actors with mode-related parameters
        self.actors = [Actor(env.state_dim, env.action_dim, num_modes=self.num_modes).to(self.device) for _ in range(self.num_agents)]
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=3e-4) for actor in self.actors]
        
        self.critic = Critic(self.num_agents, env.state_dim, env.action_dim).to(self.device)
        self.target_critic = Critic(self.num_agents, env.state_dim, env.action_dim).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

        self.memory = ReplayBuffer(100000)
        self.batch_size = 256
        self.tau = 0.005
        self.gamma = 0.99
        
        self.target_entropy = -self.env.action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
        self.alpha = self.log_alpha.exp().item()

        self.episode_rewards = []
        self.success_rates = []
        self.collision_rates = []

    # NEW: Supervisor logic to determine agent modes
    def _get_strategic_modes(self):
        modes = []
        positions = self.env.positions
        time_remaining = self.env.time_remaining
        time_limits = self.env.time_limits
        
        for i in range(self.env.num_agents):
            current_mode = AgentMode.CRUISE
            
            # RUSH logic: activated in the last 25% of allotted time
            if time_remaining[i] < time_limits[i] * 0.25:
                current_mode = AgentMode.RUSH
                
            # AVOID logic: overrides RUSH if an agent is too close
            is_proximate = False
            for j in range(self.env.num_agents):
                if i == j: continue
                dist = np.linalg.norm(positions[i] - positions[j])
                # Use a threshold slightly larger than collision radius but smaller than comms radius
                if dist < self.env.collision_radius * 4: 
                    is_proximate = True
                    break
            
            if is_proximate:
                current_mode = AgentMode.AVOID
            
            modes.append(current_mode.value)
        return torch.LongTensor(modes).to(self.device)

    # CHANGED: select_joint_action now takes modes as input
    def select_joint_action(self, states, graph_data, modes, evaluate=False):
        actions = []
        with torch.no_grad():
            for i in range(self.num_agents):
                state_tensor = torch.FloatTensor(states[i]).unsqueeze(0).to(self.device)
                mode_tensor = modes[i].unsqueeze(0) # Get the mode for this agent
                
                agent_graph_data = Data(
                    x=graph_data.x, 
                    edge_index=graph_data.edge_index, 
                    agent_index=i
                ).to(self.device)
                
                if evaluate:
                    mean, _ = self.actors[i](state_tensor, mode_tensor, agent_graph_data)
                    action = torch.tanh(mean).cpu().numpy().flatten()
                else:
                    action, _ = self.actors[i].sample(state_tensor, mode_tensor, agent_graph_data)
                    action = action.cpu().numpy().flatten()
                
                actions.append(action)
        return np.array(actions)

    def train(self):
        for episode in range(self.num_episodes):
            self.env.current_episode = episode
            states = self.env.reset()
            episode_reward = 0
            had_collision = False
            
            for step in range(self.env.max_episode_steps):
                edge_index = self.env.get_communication_graph().to(self.device)
                node_features = torch.FloatTensor(states).to(self.device)
                graph_data = Data(x=node_features, edge_index=edge_index)
                
                # NEW: Get modes from the supervisor before acting
                modes = self._get_strategic_modes()
                
                # CHANGED: Pass modes to the action selection
                actions = self.select_joint_action(states, graph_data, modes, evaluate=False)
                next_states, reward, done, _ = self.env.step(actions)
                
                next_edge_index = self.env.get_communication_graph().to(self.device)
                
                # CHANGED: Store modes in the replay buffer
                self.memory.push(
                    states, actions, reward, next_states, done,
                    edge_index.cpu(), next_edge_index.cpu(), modes.cpu().numpy()
                )
                
                states = next_states
                episode_reward += reward
                
                for i in range(self.num_agents):
                    for j in range(i + 1, self.num_agents):
                        if np.linalg.norm(self.env.positions[i] - self.env.positions[j]) < self.env.collision_radius:
                            had_collision = True
                
                if len(self.memory) > self.batch_size * 5:
                    self.update_centralized()

                if done:
                    break
            
            distances = np.linalg.norm(self.env.positions - self.env.goals, axis=1)
            success = 1.0 if np.all(distances < self.env.goal_radius) else 0.0
            
            self.episode_rewards.append(episode_reward)
            self.success_rates.append(success)
            self.collision_rates.append(1.0 if had_collision else 0.0)

            if episode > 0 and episode % 50 == 0:
                avg_reward = np.mean(self.episode_rewards[-50:])
                avg_success = np.mean(self.success_rates[-50:])
                avg_collision = np.mean(self.collision_rates[-50:])
                print(f"Ep {episode:4d} | Reward: {avg_reward:7.2f} | Success: {avg_success:.3f} | Collision: {avg_collision:.3f} | α: {self.alpha:.3f}")

    def update_centralized(self):
        batch = self.memory.sample(self.batch_size)
        
        joint_states = torch.FloatTensor(np.array([t.joint_state for t in batch])).to(self.device)
        joint_actions = torch.FloatTensor(np.array([t.joint_action for t in batch])).to(self.device)
        rewards = torch.FloatTensor([t.reward for t in batch]).unsqueeze(1).to(self.device)
        joint_next_states = torch.FloatTensor(np.array([t.joint_next_state for t in batch])).to(self.device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.bool).unsqueeze(1).to(self.device)
        # NEW: Load modes from the replay buffer
        modes = torch.LongTensor(np.array([t.modes for t in batch])).to(self.device)
        
        with torch.no_grad():
            next_actions_list, log_probs_list = [], []
            for i in range(self.num_agents):
                # CHANGED: Pass the mode to the actor when sampling next actions
                action, log_prob = self.actors[i].sample(joint_next_states[:, i, :], modes[:, i], graph_data=None)
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
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # Actor update
        new_actions_list, log_probs_list = [], []
        for i in range(self.num_agents):
            # CHANGED: Pass the mode to the actor for the policy update step
            new_actions, log_probs = self.actors[i].sample(joint_states[:, i, :], modes[:, i], graph_data=None)
            new_actions_list.append(new_actions)
            log_probs_list.append(log_probs)
        
        joint_new_actions = torch.stack(new_actions_list, dim=1)
        joint_log_probs = torch.cat(log_probs_list, dim=1).sum(dim=1, keepdim=True)

        q1_new, q2_new = self.critic(joint_states, joint_new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * joint_log_probs - q_new).mean()

        # This part requires careful handling. Since all actors contribute to one loss,
        # we can compute the gradient once and then step each optimizer.
        for optim in self.actor_optimizers:
            optim.zero_grad()
        actor_loss.backward()
        for i, optim in enumerate(self.actor_optimizers):
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 1.0)
            optim.step()
        
        # Temperature update
        alpha_loss = -(self.log_alpha.exp() * (joint_log_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item()

        # Soft update target networks
        with torch.no_grad():
            for target, source in zip(self.target_critic.parameters(), self.critic.parameters()):
                target.data.mul_(1.0 - self.tau)
                target.data.add_(self.tau * source.data)
                
    def plot_training_curves(self):
        # This function remains unchanged
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        fig.suptitle('Training Progress', fontsize=16)
        window = 50

        if len(self.episode_rewards) >= window:
            avg_rewards = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
            avg_success = np.convolve(self.success_rates, np.ones(window)/window, mode='valid')
            avg_collision = np.convolve(self.collision_rates, np.ones(window)/window, mode='valid')
            episodes = np.arange(window - 1, len(self.episode_rewards))

            axes[0].plot(episodes, avg_rewards, color='tab:blue', linewidth=2)
            axes[0].set_title('Average Episode Reward (50-ep MA)')
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(episodes, avg_success, color='tab:green', linewidth=2)
            axes[1].set_title('Success Rate (50-ep MA)')
            axes[1].set_ylim([0, 1.05])
            axes[1].grid(True, alpha=0.3)

            axes[2].plot(episodes, avg_collision, color='tab:red', linewidth=2)
            axes[2].set_title('Collision Rate (50-ep MA)')
            axes[2].set_xlabel('Episode')
            axes[2].set_ylim([0, 1.05])
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig('training_curves_hybrid.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Training curves saved as 'training_curves_hybrid.png'")
        
# --- Main Execution ---

if __name__ == "__main__":
    env = MultiAgentEnvironment(num_agents=4)
    trainer = MultiAgentTrainer(env, num_episodes=3000)
    
    print("Starting Multi-Agent RL Training with Hybrid Control...")
    print(f"Device: {trainer.device}")
    print(f"Agents: {env.num_agents}")
    print(f"State dim: {env.state_dim}, Action dim: {env.action_dim}")
    print("-" * 60)
    
    trainer.train()
    trainer.plot_training_curves()
    
    print("\nTraining completed!")
    print(f"Final 100-episode success rate: {np.mean(trainer.success_rates[-100:]):.3f}")
    print(f"Final 100-episode avg reward: {np.mean(trainer.episode_rewards[-100:]):.2f}")