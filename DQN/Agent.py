import gymnasium
import highway_env
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import random

from .Network import QNetwork
from .Replay import ReplayBuffer

class DQNAgent:
    """
    Deep Q-Network (DQN) Agent.
    - Uses Q-learning to learn an optimal policy.
    - Stores experiences in a Replay Buffer.
    - Uses a Target Network to stabilize learning.
    """
    def __init__(self, state_dim, action_dim, hidden_layers=[128, 128], 
                 lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, 
                 min_epsilon=0.01, target_update_freq=10, buffer_capacity=100000):
        """
        Initialize the DQN Agent.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.target_update_freq = target_update_freq

        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize Q-network and target Q-network
        self.q_net = QNetwork(state_dim, action_dim, hidden_layers).to(self.device)
        self.target_q_net = QNetwork(state_dim, action_dim, hidden_layers).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())  # Copy weights
        self.target_q_net.eval()  # Target network is not trained, only updated periodically

        # Define optimizer
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # Step counter for target network update
        self.step_count = 0

    def select_action(self, state):
        """
        Select an action using an epsilon-greedy strategy.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)  # Explore
        else:
            # Flatten the state to a 1D vector and add batch dimension.
            state_tensor = torch.FloatTensor(state).reshape(1, -1).to(self.device)
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
            return torch.argmax(q_values).item()  # Exploit 

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = batch

        # Ensure states are flattened to 2D: (batch_size, -1)
        states = states.to(self.device).view(states.size(0), -1)
        next_states = next_states.to(self.device).view(next_states.size(0), -1)
        
        actions = actions.to(self.device).unsqueeze(1)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        # Compute current Q-values; now states is (batch_size, state_dim)
        current_q_values = self.q_net(states).gather(1, actions).squeeze()

        # Compute target Q-values using the Bellman equation
        with torch.no_grad():
            next_q_values = self.target_q_net(next_states).max(dim=1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        return nn.MSELoss()(current_q_values, target_q_values.detach())

    def update_q_values(self, batch_size=64):
        """
        Perform Q-learning update using a batch of experiences.
        """
        if self.replay_buffer.size() < batch_size:
            return  # Skip update if not enough samples in buffer

        batch = self.replay_buffer.sample(batch_size)
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.sync_target_network()
            
        return loss.item()

    def sync_target_network(self):
        """
        Update the target network with weights from the Q-network.
        """
        self.target_q_net.load_state_dict(self.q_net.state_dict())
    
    def learn(self, env, num_episodes=100, batch_size=32, max_steps=200,
              learning_starts=200, train_freq=1, gradient_steps=1, verbose=1, logger=None):
        """
        Run the training loop for a given environment.
        :param env: Gym-like environment.
        :param num_episodes: Number of episodes (e.g., 100 episodes * 200 steps = ~20k timesteps).
        :param batch_size: Batch size for training.
        :param max_steps: Maximum steps per episode.
        :param learning_starts: Minimum steps before starting training updates.
        :param train_freq: Frequency (in steps) at which to update.
        :param gradient_steps: Number of gradient updates per training step.
        :param verbose: If >0, prints episode summary.
        :param logger: Optional TrainingLogger instance.
        """
        total_steps = 0
        for episode in range(num_episodes):
            state = env.reset()[0]  # Gymnasium returns (observation, info)
            episode_reward = 0.0
            episode_loss = 0.0
            steps = 0

            while steps < max_steps:
                action = self.select_action(state)
                next_state, reward, done, truncated, _ = env.step(action)
                self.replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                steps += 1
                total_steps += 1

                if total_steps >= learning_starts and total_steps % train_freq == 0:
                    for _ in range(gradient_steps):
                        loss = self.update_q_values(batch_size)
                        if loss is not None:
                            episode_loss += loss

                if done or truncated:
                    break

            avg_loss = episode_loss / steps if steps > 0 else 0.0
            if verbose:
                print(f"Episode {episode+1}/{num_episodes} - Reward: {episode_reward:.2f}, "
                      f"Avg Loss: {avg_loss:.4f}, Epsilon: {self.epsilon:.3f}")

            if logger:
                logger.log_episode(episode_reward, avg_loss, self.epsilon, steps)
                if episode % logger.save_freq == 0:
                    logger.save_model(self.q_net, episode)
                    logger.save_replay_buffer(self.replay_buffer)
    
    def save(self, path):
        """Save the Q-network's state."""
        torch.save(self.q_net.state_dict(), path)
    
    def load(self, path):
        """Load the Q-network's state."""
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.sync_target_network()

        