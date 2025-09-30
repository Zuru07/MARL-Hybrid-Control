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
Transition = namedtuple('Transition', ['joint_state', 'joint_action', 'reward', 'joint_next_state', 'done'])

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
        self.mlp = nn.Sequential(nn.Linear(input_dim * 2, output_dim), nn.ReLU(), nn.Linear(output_dim, output_dim))
    def forward(self, x, edge_index): return self.propagate(edge_index, x=x)
    def message(self, x_i, x_j): return self.mlp(torch.cat([x_i, x_j], dim=-1))

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

class NoisyActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, gnn_dim=64, noise_stddev=0.1):
        super(NoisyActor, self).__init__()
        self.noise_stddev = noise_stddev
        self.actor = Actor(state_dim, action_dim, hidden_dim, gnn_dim)
        self.noisy_actor = Actor(state_dim, action_dim, hidden_dim, gnn_dim)
        self.noisy_actor.load_state_dict(self.actor.state_dict())

    def parameters(self): return self.actor.parameters()
    def add_noise(self):
        with torch.no_grad():
            for param, noisy_param in zip(self.actor.parameters(), self.noisy_actor.parameters()):
                noise = torch.randn_like(param.data) * self.noise_stddev
                noisy_param.data.copy_(param.data + noise)
    
    def sample(self, state, graph_data=None): return self.actor.sample(state, graph_data)
    def sample_noisy(self, state, graph_data=None):
        mean, log_std = self.noisy_actor(state, graph_data)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        return action
    def forward(self, state, graph_data=None): return self.actor(state, graph_data)

class Critic(nn.Module):
    def __init__(self, num_agents, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        input_dim = num_agents * (state_dim + action_dim)
        self.q1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.q2 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
    def forward(self, all_states, all_actions):
        x = torch.cat([all_states, all_actions], dim=-1)
        x = x.view(x.size(0), -1)
        return self.q1(x), self.q2(x)

# --- Simulation Environment ---

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
        if self.current_episode < 1000: self.goal_radius = 1.5; time_buffer = 10.0
        elif self.current_episode < 2000: self.goal_radius = 0.8; time_buffer = 6.0
        else: self.goal_radius = 0.3; time_buffer = 3.0
        
        self.positions = np.random.uniform(-self.world_size / 2, self.world_size / 2, (self.num_agents, 2))
        self.velocities = np.zeros((self.num_agents, 2))
        self.goals = np.random.uniform(-self.world_size / 2, self.world_size / 2, (self.num_agents, 2))
        
        distances_to_goal = np.linalg.norm(self.goals - self.positions, axis=1)
        self.time_limits = distances_to_goal / (self.max_speed * 0.8) + time_buffer
        self.time_remaining = self.time_limits.copy()
        
        self.step_count = 0
        self.done = False
        
        return self.get_observations()
    
    def get_observations(self):
        observations = []
        for i in range(self.num_agents):
            pos_norm = self.positions[i] / (self.world_size / 2)
            vel_norm = self.velocities[i] / self.max_speed
            time_norm = np.clip(self.time_remaining[i] / self.time_limits[i], 0, 1) * 2 - 1
            obs = np.concatenate([pos_norm, vel_norm, [time_norm]])
            observations.append(obs)
        return np.array(observations)
    
    def get_communication_graph(self):
        pos_diff = self.positions[:, np.newaxis, :] - self.positions[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(pos_diff, axis=-1)
        adj_matrix = (dist_matrix > 0) & (dist_matrix <= self.communication_radius)
        edge_indices = np.array(np.where(adj_matrix)).T
        if edge_indices.shape[0] == 0: return torch.empty((2, 0), dtype=torch.long)
        return torch.LongTensor(edge_indices).t().contiguous()
    
    def step(self, actions):
        actions = np.clip(actions, -1, 1)
        self.velocities += actions * self.dt
        
        speeds = np.linalg.norm(self.velocities, axis=1)
        mask = speeds > self.max_speed
        if np.any(mask): self.velocities[mask] *= self.max_speed / speeds[mask, np.newaxis]
        
        self.positions += self.velocities * self.dt
        self.positions = np.clip(self.positions, -self.world_size / 2, self.world_size / 2)
        
        self.time_remaining -= self.dt
        self.step_count += 1
        
        reward = self.calculate_rewards()
        self.check_termination()
        
        return self.get_observations(), reward, self.done, {}
    
    def calculate_rewards(self):
        """Final reward function to incentivize coordination and task completion."""
        team_reward = 0.0
        
        success_bonus = 250.0
        collision_penalty = -10.0
        step_penalty = -0.01
        distance_penalty_factor = -0.1
        # --- CHANGE START: New reward for waiting at the goal ---
        staging_reward = 2.5
        # --- CHANGE END ---
        
        team_reward += step_penalty
        num_collisions = 0
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if np.linalg.norm(self.positions[i] - self.positions[j]) < self.collision_radius:
                    num_collisions += 1
        team_reward += collision_penalty * num_collisions
        
        current_dist_to_goal = np.linalg.norm(self.positions - self.goals, axis=1)
        team_reward += np.sum(current_dist_to_goal) * distance_penalty_factor
        
        # --- CHANGE START: Sustained reward for waiting at goal ---
        num_at_goal = 0
        for i in range(self.num_agents):
            if current_dist_to_goal[i] < self.goal_radius:
                num_at_goal += 1
                team_reward += staging_reward # Give a small reward for each step an agent is at its goal
        # --- CHANGE END ---
        
        if num_at_goal == self.num_agents:
            team_reward += success_bonus
            self.done = True

        if np.any(self.time_remaining <= 0):
            # Penalize based on how many agents failed to reach the goal
            team_reward -= (self.num_agents - num_at_goal) * 25.0
        
        return team_reward

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

        self.actors = [NoisyActor(env.state_dim, env.action_dim).to(self.device) for _ in range(self.num_agents)]
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=3e-4) for actor in self.actors]
        
        self.critic = Critic(self.num_agents, env.state_dim, env.action_dim).to(self.device)
        self.target_critic = Critic(self.num_agents, env.state_dim, env.action_dim).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

        self.memory = ReplayBuffer(100000)

        self.batch_size = 512
        self.tau = 0.001
        self.gamma = 0.99
        
        self.target_entropy = -self.env.action_dim * self.num_agents
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
        self.alpha = self.log_alpha.exp().item()

        self.episode_rewards = []
        self.success_rates = []

    def select_joint_action(self, states, graph_data, evaluate=False):
        """Selects a joint action, augmenting learned policy with a refined hybrid supervisor."""
        actions = []
        with torch.no_grad():
            for i in range(self.num_agents):
                state_tensor = torch.FloatTensor(states[i]).unsqueeze(0).to(self.device)
                agent_graph_data = Data(x=graph_data.x, edge_index=graph_data.edge_index, agent_index=i).to(self.device)
                
                if evaluate:
                    mean, _ = self.actors[i](state_tensor, agent_graph_data)
                    actor_action = torch.tanh(mean).cpu().numpy().flatten()
                else:
                    action_tensor = self.actors[i].sample_noisy(state_tensor, agent_graph_data)
                    actor_action = action_tensor.cpu().numpy().flatten()

                # --- CHANGE START: Refined Supervisor with Nudging ---
                if not evaluate:
                    final_action = actor_action.copy()
                    # Rule 1: Emergency collision avoidance (nudging)
                    min_dist_to_ally = float('inf')
                    nearest_ally_pos = None
                    for j in range(self.num_agents):
                        if i != j:
                            dist = np.linalg.norm(self.env.positions[i] - self.env.positions[j])
                            if dist < min_dist_to_ally:
                                min_dist_to_ally = dist
                                nearest_ally_pos = self.env.positions[j]

                    if min_dist_to_ally < self.env.collision_radius * 2.0: # Increased safety margin
                        avoid_vec = self.env.positions[i] - nearest_ally_pos
                        avoid_action = avoid_vec / (np.linalg.norm(avoid_vec) + 1e-6)
                        # Nudge the original action away from collision
                        final_action += avoid_action * 0.5 
                else:
                    final_action = actor_action
                # --- CHANGE END ---
                
                actions.append(final_action)
        return np.array(actions)

    def train(self):
        for episode in range(self.num_episodes):
            self.env.current_episode = episode
            states = self.env.reset()
            episode_reward = 0
            
            for actor in self.actors: actor.add_noise()
            
            for step in range(self.env.max_episode_steps):
                edge_index = self.env.get_communication_graph().to(self.device)
                node_features = torch.FloatTensor(states).to(self.device)
                graph_data = Data(x=node_features, edge_index=edge_index)
                
                actions = self.select_joint_action(states, graph_data)
                next_states, reward, done, _ = self.env.step(actions)
                
                self.memory.push(states, actions, reward, next_states, done)
                
                states = next_states
                episode_reward += reward
                
                if len(self.memory) > self.batch_size * 10:
                    self.update_centralized()

                if done: break
            
            num_successful = np.sum(np.linalg.norm(self.env.positions - self.env.goals, axis=1) < self.env.goal_radius)
            success = 1.0 if num_successful == self.num_agents else 0.0
            
            self.episode_rewards.append(episode_reward)
            self.success_rates.append(success)

            if episode > 0 and episode % 50 == 0:
                avg_reward = np.mean(self.episode_rewards[-50:])
                avg_success = np.mean(self.success_rates[-50:])
                print(f"Episode {episode}, Avg Reward (last 50): {avg_reward:.2f}, Avg Team Success: {avg_success:.2f}")

    def update_centralized(self):
        batch = self.memory.sample(self.batch_size)
        
        joint_states = torch.FloatTensor(np.array([t.joint_state for t in batch])).to(self.device)
        joint_actions = torch.FloatTensor(np.array([t.joint_action for t in batch])).to(self.device)
        rewards = torch.FloatTensor([t.reward for t in batch]).unsqueeze(1).to(self.device)
        joint_next_states = torch.FloatTensor(np.array([t.joint_next_state for t in batch])).to(self.device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.bool).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            next_actions_list, log_probs_list = [], []
            for i in range(self.num_agents):
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

        new_actions_list, log_probs_list = [], []
        for i in range(self.num_agents):
            new_actions, log_probs = self.actors[i].sample(joint_states[:, i, :], graph_data=None)
            new_actions_list.append(new_actions)
            log_probs_list.append(log_probs)
        
        joint_new_actions = torch.stack(new_actions_list, dim=1)
        joint_log_probs = torch.cat(log_probs_list, dim=1).sum(dim=1, keepdim=True)

        q1_new, q2_new = self.critic(joint_states, joint_new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * joint_log_probs - q_new).mean()

        for optim in self.actor_optimizers: optim.zero_grad()
        actor_loss.backward()
        for optim in self.actor_optimizers: optim.step()
        
        alpha_loss = -(self.log_alpha.exp() * (joint_log_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item()

        with torch.no_grad():
            for target, source in zip(self.target_critic.parameters(), self.critic.parameters()):
                target.data.mul_(1.0 - self.tau)
                target.data.add_(self.tau * source.data)
                
    def plot_training_curves(self):
        fig, axes = plt.subplots(2, 1, figsize=(16, 12))
        fig.suptitle('Training Progress', fontsize=16)
        window = 50

        if len(self.episode_rewards) >= window:
            avg_rewards = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
            avg_success = np.convolve(self.success_rates, np.ones(window)/window, mode='valid')
            episodes = np.arange(window - 1, len(self.episode_rewards))

            axes[0].plot(episodes, avg_rewards, color='tab:blue')
            axes[0].set_title('Average Reward (50-Episode Rolling Average)')
            axes[0].set_xlabel('Episode')
            axes[0].set_ylabel('Average Reward')
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(episodes, avg_success, color='tab:green')
            axes[1].set_title('Average Success Rate (50-Episode Rolling Average)')
            axes[1].set_xlabel('Episode')
            axes[1].set_ylabel('Average Success Rate')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig('training_curves.png', dpi=150)
        plt.show()

# --- Main Execution ---

if __name__ == "__main__":
    env = MultiAgentEnvironment(num_agents=4)
    trainer = MultiAgentTrainer(env, num_episodes=3000)
    
    print("Starting training...")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    trainer.train()
    trainer.plot_training_curves()
    
    print("Training completed!")