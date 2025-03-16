import json
import torch
import pickle
import os

import numpy as np

class TrainingLogger:
    """
    Handles logging and saving training data for RL models.
    """
    def __init__(self, log_dir="logs", save_freq=50):
        """
        Initialize the logger.
        :param log_dir: Directory to save logs and model checkpoints.
        :param save_freq: Number of episodes between model saves.
        """
        self.log_dir = log_dir
        self.save_freq = save_freq

        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)

        # Store training history
        self.episode_rewards = []
        self.episode_losses = []
        self.epsilon_values = []
        self.steps_per_episode = []

    def log_episode(self, reward, loss, epsilon, steps):
        """
        Logs episode data.
        :param reward: Total reward in the episode.
        :param loss: Average loss for the episode.
        :param epsilon: Epsilon value at the end of the episode.
        :param steps: Total steps in the episode.
        :param action_qvalues: List of (action, max Q-value) pairs for analysis.
        """
        self.episode_rewards.append(reward)
        self.episode_losses.append(loss)
        self.epsilon_values.append(epsilon)
        self.steps_per_episode.append(steps)

    def save_logs(self):
        """
        Saves training logs as a JSON file.
        """
        log_data = {
            "rewards": self.episode_rewards,
            "losses": self.episode_losses,
            "epsilon": self.epsilon_values,
            "steps_per_episode": self.steps_per_episode
        }
        with open(os.path.join(self.log_dir, "training_logs.json"), "w") as f:
            json.dump(log_data, f, indent=4)

    def save_model(self, model, episode):
        """
        Saves the model checkpoint.
        :param model: The Q-network to save.
        :param episode: The current episode (used in filename).
        """
        model_path = os.path.join(self.log_dir, f"checkpoint_{episode}.pth")
        torch.save(model.state_dict(), model_path)

    def save_replay_buffer(self, replay_buffer):
        """
        Saves the replay buffer as a pickle file.
        :param replay_buffer: The replay buffer object.
        """
        buffer_path = os.path.join(self.log_dir, "replay_buffer.pkl")
        with open(buffer_path, "wb") as f:
            pickle.dump(replay_buffer.buffer, f)

### Test ###
def test():
    """
    Test TrainingLogger by simulating training logs and model saving.
    """
    log_dir = "test_logs"
    logger = TrainingLogger(log_dir=log_dir, save_freq=5)

    # Simulated training parameters
    state_dim = 4
    action_dim = 2
    replay_buffer = ReplayBuffer(capacity=1000)

    # Create a test Q-network
    model = QNetwork(state_dim, action_dim, hidden_layers=[64, 128])

    for episode in range(20):  # Simulate 20 episodes
        reward = np.random.randint(10, 100)
        loss = np.random.random()
        epsilon = np.random.random()
        steps = np.random.randint(10, 200)

        # Log the episode data
        logger.log_episode(reward, loss, epsilon, steps)

        # Simulate adding data to replay buffer
        for _ in range(20):  # Add 20 experiences per episode
            state = np.random.rand(state_dim)
            action = np.random.randint(0, action_dim)
            next_state = np.random.rand(state_dim)
            done = np.random.choice([True, False])
            replay_buffer.add(state, action, reward, next_state, done)

        # Save model and replay buffer every `save_freq` episodes
        if episode % logger.save_freq == 0:
            logger.save_model(model, episode)
            logger.save_replay_buffer(replay_buffer)

    # Save logs at the end
    logger.save_logs()

    print("Test completed. Check 'test_logs/' for saved files.")

# Run the test
if __name__ == "__main__":
    from Replay import ReplayBuffer
    from Network import QNetwork
    test()
