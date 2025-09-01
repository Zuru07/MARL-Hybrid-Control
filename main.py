import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.data import Data, Batch
import matplotlib.pyplot as plt
import random
from collections import deque, namedtuple
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Experience replay buffer
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, *args):
        self.buffer.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# Graph Neural Network for agent communication
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
        
    def forward(self, x, edge_index, batch=None):
        x = F.relu(self.gnn1(x, edge_index))
        x = self.gnn2(x, edge_index)
        return x

# Actor Network (Policy)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, gnn_dim=64):
        super(Actor, self).__init__()
        self.graph_encoder = GraphEncoder(state_dim, hidden_dim//2, gnn_dim)
        
        # Combine individual state with graph embedding
        self.fc1 = nn.Linear(state_dim + gnn_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
        self.max_log_std = 2
        self.min_log_std = -20
    
    def forward(self, state, graph_data=None):
        if graph_data is not None:
            # Extract graph features
            graph_embed = self.graph_encoder(graph_data.x, graph_data.edge_index, graph_data.batch)
            # Aggregate graph embedding for each agent
            if hasattr(graph_data, 'batch') and graph_data.batch is not None:
                from torch_scatter import scatter_mean
                graph_embed = scatter_mean(graph_embed, graph_data.batch, dim=0)
            else:
                graph_embed = graph_embed.mean(dim=0, keepdim=True)
        else:
            graph_embed = torch.zeros(state.shape[0], 64, device=state.device)
        
        # Combine state with graph embedding
        x = torch.cat([state, graph_embed], dim=-1)
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
        return action, log_prob, mean

# Critic Network (Q-function)
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, gnn_dim=64):
        super(Critic, self).__init__()
        self.graph_encoder = GraphEncoder(state_dim + action_dim, hidden_dim//2, gnn_dim)
        
        # Two Q-networks for Double Q-learning
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim + gnn_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim + gnn_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action, graph_data=None):
        if graph_data is not None:
            # Combine state and action for graph processing
            node_features = torch.cat([graph_data.x, action.repeat_interleave(graph_data.x.size(0) // action.size(0), dim=0)], dim=-1)
            graph_embed = self.graph_encoder(node_features, graph_data.edge_index, graph_data.batch)
            
            if hasattr(graph_data, 'batch') and graph_data.batch is not None:
                from torch_scatter import scatter_mean
                graph_embed = scatter_mean(graph_embed, graph_data.batch, dim=0)
            else:
                graph_embed = graph_embed.mean(dim=0, keepdim=True)
        else:
            graph_embed = torch.zeros(state.shape[0], 64, device=state.device)
        
        x = torch.cat([state, action, graph_embed], dim=-1)
        q1 = self.q1(x)
        q2 = self.q2(x)
        return q1, q2

# Multi-Agent Environment
class MultiAgentEnvironment:
    def __init__(self, num_agents=4, world_size=10.0, communication_radius=3.0, max_episode_steps=100):
        self.num_agents = num_agents
        self.world_size = world_size
        self.communication_radius = communication_radius
        self.max_episode_steps = max_episode_steps
        
        # Agent properties
        self.state_dim = 5  # [x, y, vx, vy, time_remaining]
        self.action_dim = 2  # [ax, ay] - acceleration
        
        # Environment parameters
        self.max_speed = 2.0
        self.dt = 0.1
        self.collision_radius = 0.5
        self.goal_radius = 0.3
        
        self.reset()
    
    def reset(self):
        # Initialize agent positions randomly
        self.positions = np.random.uniform(-self.world_size/2, self.world_size/2, (self.num_agents, 2))
        self.velocities = np.zeros((self.num_agents, 2))
        
        # Random goal positions
        self.goals = np.random.uniform(-self.world_size/2, self.world_size/2, (self.num_agents, 2))
        
        # Ensure agents and goals are not too close initially
        while True:
            valid = True
            for i in range(self.num_agents):
                for j in range(i+1, self.num_agents):
                    if np.linalg.norm(self.positions[i] - self.positions[j]) < self.collision_radius * 2:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                break
            self.positions = np.random.uniform(-self.world_size/2, self.world_size/2, (self.num_agents, 2))
        
        # Time constraints
        distances_to_goal = np.linalg.norm(self.goals - self.positions, axis=1)
        self.time_limits = distances_to_goal / (self.max_speed * 0.7) + 20  # Add buffer time
        self.time_remaining = self.time_limits.copy()
        
        self.step_count = 0
        self.done = np.zeros(self.num_agents, dtype=bool)
        
        return self.get_observations()
    
    def get_observations(self):
        observations = []
        for i in range(self.num_agents):
            obs = np.concatenate([
                self.positions[i],
                self.velocities[i],
                [self.time_remaining[i]]
            ])
            observations.append(obs)
        return np.array(observations)
    
    def get_communication_graph(self):
        # Build communication graph based on proximity
        edge_indices = []
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if i != j:
                    distance = np.linalg.norm(self.positions[i] - self.positions[j])
                    if distance <= self.communication_radius:
                        edge_indices.append([i, j])
        
        if len(edge_indices) == 0:
            # If no connections, create self-loops
            edge_indices = [[i, i] for i in range(self.num_agents)]
        
        return torch.LongTensor(edge_indices).t().contiguous()
    
    def step(self, actions):
        # Apply actions (accelerations)
        actions = np.clip(actions, -1, 1)  # Limit acceleration
        
        # Update velocities
        self.velocities += actions * self.dt
        # Limit velocities
        speeds = np.linalg.norm(self.velocities, axis=1)
        mask = speeds > self.max_speed
        self.velocities[mask] = self.velocities[mask] / speeds[mask, np.newaxis] * self.max_speed
        
        # Update positions
        self.positions += self.velocities * self.dt
        
        # Keep agents within world boundaries
        self.positions = np.clip(self.positions, -self.world_size/2, self.world_size/2)
        
        # Update time
        self.time_remaining -= self.dt
        self.step_count += 1
        
        # Calculate rewards and check termination
        rewards = self.calculate_rewards()
        self.check_termination()
        
        return self.get_observations(), rewards, self.done.copy(), {}
    
    def calculate_rewards(self):
    # This function now calculates ONE team reward
        team_reward = 0.0
        
        # --- Penalty Section ---
        # Small penalty for every step to encourage speed
        team_reward -= 0.01 
        
        # Collision penalty
        num_collisions = 0
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents): # Avoid double counting
                dist = np.linalg.norm(self.positions[i] - self.positions[j])
                if dist < self.collision_radius:
                    num_collisions += 1
        team_reward -= 2.0 * num_collisions # One penalty per collision pair
        
        # --- Reward Section ---
        all_agents_succeeded = True
        for i in range(self.num_agents):
            if self.done[i]: # Skip agents that have already finished
                continue

            dist_to_goal = np.linalg.norm(self.positions[i] - self.goals[i])
            
            # Reward for getting closer (negative distance)
            team_reward += -0.1 * dist_to_goal 

            # Big reward for reaching goal
            if dist_to_goal < self.goal_radius:
                team_reward += 10.0
                self.done[i] = True
            else:
                all_agents_succeeded = False # At least one agent is not done

            # Big penalty for running out of time
            if self.time_remaining[i] <= 0:
                team_reward -= 5.0
                self.done[i] = True

        # Final bonus for full team success
        if all_agents_succeeded:
            team_reward += 20.0

        # Return the SAME reward value to every agent
        return np.full(self.num_agents, team_reward/self.num_agents)
    
    def check_termination(self):
        # Check if episode should end
        for i in range(self.num_agents):
            if self.time_remaining[i] <= 0:
                self.done[i] = True
        
        # Episode ends if all agents are done or max steps reached
        if np.all(self.done) or self.step_count >= self.max_episode_steps:
            self.done[:] = True
    
    def render(self, save_path=None):
        plt.figure(figsize=(10, 10))
        plt.xlim(-self.world_size/2, self.world_size/2)
        plt.ylim(-self.world_size/2, self.world_size/2)
        
        # Draw agents
        for i in range(self.num_agents):
            color = f'C{i}'
            plt.scatter(self.positions[i, 0], self.positions[i, 1], 
                       s=100, c=color, marker='o', label=f'Agent {i}')
            plt.scatter(self.goals[i, 0], self.goals[i, 1], 
                       s=100, c=color, marker='*', alpha=0.5)
            
            # Draw velocity vector
            plt.arrow(self.positions[i, 0], self.positions[i, 1],
                     self.velocities[i, 0], self.velocities[i, 1],
                     head_width=0.1, head_length=0.1, fc=color, ec=color, alpha=0.7)
        
        plt.title(f'Multi-Agent Environment (Step: {self.step_count})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

# SAC Agent with Graph Networks
class SACAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, tau=0.005, alpha=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.target_critic = Critic(state_dim, action_dim).to(self.device)
        
        # Copy parameters to target network
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Hyperparameters
        self.tau = tau
        self.alpha = alpha
        self.gamma = 0.99
        
        # Replay buffer
        self.memory = ReplayBuffer(100000)
    
    def select_action(self, state, graph_data=None, evaluate=False):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if evaluate:
                mean, _ = self.actor(state_tensor, graph_data)
                action = torch.tanh(mean)
            else:
                action, _, _ = self.actor.sample(state_tensor, graph_data)
            return action.cpu().numpy()[0]
    
    def update(self, batch_size=256):
        if len(self.memory) < batch_size:
            return {}
        
        batch = self.memory.sample(batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor([t.state for t in batch]).to(self.device)
        actions = torch.FloatTensor([t.action for t in batch]).to(self.device)
        rewards = torch.FloatTensor([t.reward for t in batch]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor([t.next_state for t in batch]).to(self.device)
        dones = torch.tensor([bool(t.done) for t in batch], dtype=torch.bool).unsqueeze(1).to(self.device)

        
        # Update Critic
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            q1_next, q2_next = self.target_critic(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = rewards + self.gamma * q_next * (~dones)
        
        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update Actor
        new_actions, log_probs, _ = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item()
        }

# Training Loop
class MultiAgentTrainer:
    def __init__(self, env, num_episodes):
        self.env = env
        self.num_episodes = num_episodes
        self.agents = [SACAgent(env.state_dim, env.action_dim) for _ in range(env.num_agents)]
        
        # Metrics
        self.episode_rewards = []
        self.success_rates = []
        self.collision_rates = []
    
    def train(self):
        for episode in range(self.num_episodes):
            states = self.env.reset()
            episode_reward = 0
            
            while not np.all(self.env.done):
                # Get communication graph
                edge_index = self.env.get_communication_graph().to(self.agents[0].device)
                
                # Create graph data for GNN
                node_features = torch.FloatTensor(states).to(self.agents[0].device)
                graph_data = Data(x=node_features, edge_index=edge_index)
                
                # Select actions for all agents
                actions = []
                for i, agent in enumerate(self.agents):
                    if not self.env.done[i]:
                        action = agent.select_action(states[i], graph_data)
                    else:
                        action = np.zeros(self.env.action_dim)
                    actions.append(action)
                
                actions = np.array(actions)
                
                # Environment step
                next_states, rewards, dones, _ = self.env.step(actions)
                
                # Store experiences
                for i, agent in enumerate(self.agents):
                    if not self.env.done[i]:
                        agent.memory.push(states[i], actions[i], rewards[i], 
                                        next_states[i], dones[i])
                
                states = next_states
                episode_reward += np.sum(rewards)
                
                # Update agents
                if episode > 10:  # Start training after some episodes
                    for agent in self.agents:
                        agent.update()
            
            # Calculate metrics
            success_rate = np.mean([np.linalg.norm(self.env.positions[i] - self.env.goals[i]) < self.env.goal_radius 
                                  for i in range(self.env.num_agents)])
            
            self.episode_rewards.append(episode_reward)
            self.success_rates.append(success_rate)
            
            if episode % 100 == 0:
                print(f"Episode {episode}, Reward: {episode_reward:.2f}, Success Rate: {success_rate:.2f}")
                
                # Render environment occasionally
                if episode % 100 == 0:
                    self.env.render(f'episode_{episode}.png')
    
    def evaluate(self, num_episodes=10):
        """Evaluate the trained agents"""
        total_rewards = []
        success_rates = []
        
        for episode in range(num_episodes):
            states = self.env.reset()
            episode_reward = 0
            
            while not np.all(self.env.done):
                edge_index = self.env.get_communication_graph().to(self.agents[0].device)
                node_features = torch.FloatTensor(states).to(self.agents[0].device)
                graph_data = Data(x=node_features, edge_index=edge_index)
                
                actions = []
                for i, agent in enumerate(self.agents):
                    if not self.env.done[i]:
                        action = agent.select_action(states[i], graph_data, evaluate=True)
                    else:
                        action = np.zeros(self.env.action_dim)
                    actions.append(action)
                
                actions = np.array(actions)
                next_states, rewards, dones, _ = self.env.step(actions)
                states = next_states
                episode_reward += np.sum(rewards)
            
            success_rate = np.mean([np.linalg.norm(self.env.positions[i] - self.env.goals[i]) < self.env.goal_radius 
                                  for i in range(self.env.num_agents)])
            
            total_rewards.append(episode_reward)
            success_rates.append(success_rate)
        
        print(f"\nEvaluation Results:")
        print(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
        print(f"Success Rate: {np.mean(success_rates):.2f} ± {np.std(success_rates):.2f}")
        
        return total_rewards, success_rates
    
    def plot_training_curves(self):
        """Plot training metrics"""
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(self.episode_rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        
        plt.subplot(1, 3, 2)
        plt.plot(self.success_rates)
        plt.title('Success Rate')
        plt.xlabel('Episode')
        plt.ylabel('Success Rate')
        
        # Running average
        window = 50
        if len(self.episode_rewards) > window:
            running_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
            plt.subplot(1, 3, 3)
            plt.plot(running_avg)
            plt.title(f'Running Average Reward (window={window})')
            plt.xlabel('Episode')
            plt.ylabel('Average Reward')
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
        plt.show()

# Main execution
if __name__ == "__main__":
    # Create environment
    env = MultiAgentEnvironment(num_agents=4, world_size=10.0, communication_radius=3.0)
    
    # Create trainer
    trainer = MultiAgentTrainer(env, num_episodes=2000)
    
    print("Starting training...")
    print(f"Environment: {env.num_agents} agents, {env.state_dim}D state, {env.action_dim}D action")
    print(f"Device: {trainer.agents[0].device}")
    
    # Train the agents
    trainer.train()
    
    # Plot training curves
    trainer.plot_training_curves()
    
    # Evaluate trained agents
    print("\nEvaluating trained agents...")
    trainer.evaluate(num_episodes=100)
    
    # Create a final demonstration
    print("\nCreating final demonstration...")
    states = env.reset()
    env.render('final_demo_start.png')
    
    step = 0
    while not np.all(env.done) and step < 100:
        edge_index = env.get_communication_graph().to(trainer.agents[0].device)
        node_features = torch.FloatTensor(states).to(trainer.agents[0].device)
        graph_data = Data(x=node_features, edge_index=edge_index)
        
        actions = []
        for i, agent in enumerate(trainer.agents):
            if not env.done[i]:
                action = agent.select_action(states[i], graph_data, evaluate=True)
            else:
                action = np.zeros(env.action_dim)
            actions.append(action)
        
        actions = np.array(actions)
        next_states, rewards, dones, _ = env.step(actions)
        states = next_states
        step += 1
    
    env.render('final_demo_end.png')
    print("Training completed! Check the generated plots and demonstration images.")