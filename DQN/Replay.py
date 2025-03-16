import random
import numpy as np
from collections import deque
import torch

class ReplayBuffer:
    """
    Replay Buffer to store and sample past experiences.
    Helps stabilize learning by breaking correlation between consecutive experiences.
    """

    def __init__(self, capacity):
        """
        Initialize the Replay Buffer.
        :param capacity: Maximum number of experiences to store.
        """
        self.buffer = deque(maxlen=capacity)  # FIFO Queue

    def add(self, state, action, reward, next_state, done):
        """
        Store an experience in the buffer.
        :param state: Current state (numpy array or tensor)
        :param action: Action taken (int or numpy array)
        :param reward: Reward received (float)
        :param next_state: Next state after action (numpy array or tensor)
        :param done: Whether the episode ended (bool)
        """
        self.buffer.append((state, action, reward, next_state, float(done)))

    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer.
        :param batch_size: Number of samples to return.
        :return: Batch of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)  # Randomly sample experiences
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to PyTorch tensors for training
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones))

        return states, actions, rewards, next_states, dones

    def size(self):
        """
        Return the current size of the buffer.
        """
        return len(self.buffer)

def test():
    # Create a replay buffer with capacity 100
    buffer = ReplayBuffer(capacity=100)

    # Simulate dummy experiences
    for i in range(10):
        state = [i, i+1, i+2]
        action = i % 3
        reward = i * 0.5
        next_state = [i+1, i+2, i+3]
        done = i % 5 == 0  # Every 5 steps, mark as done

        buffer.add(state, action, reward, next_state, done)

    print(f"Buffer Size: {buffer.size()}")

    # Sample a batch
    batch_size = 4
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)

    print("\nSampled Batch:")
    print("States:", states)
    print("Actions:", actions)
    print("Rewards:", rewards)
    print("Next States:", next_states)
    print("Dones:", dones)

if __name__ == "__main__":
    test()
