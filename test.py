import gymnasium
import highway_env
import numpy as np

from DQN.Agent import DQNAgent  # Assuming your DQNAgent class is in Agent.py

def test_model():
    # Create the environment.
    env = gymnasium.make("highway-fast-v0", render_mode='rgb_array')
    
    # Compute the flattened state dimension.
    state_dim = int(np.prod(env.observation_space.shape))
    action_dim = env.action_space.n

    # Initialize the agent with the same hyperparameters used during training.
    agent = DQNAgent(state_dim, action_dim, hidden_layers=[256, 256],
                      lr=5e-4,
                      gamma=0.8,
                      epsilon=1.0,
                      epsilon_decay=0.995,
                      min_epsilon=0.01,
                      target_update_freq=50,
                      buffer_capacity=15000)
    
    # Load the saved model weights.
    agent.load("highway_dqn/model.pth")
    
    # Set epsilon to 0 for deterministic action selection during testing.
    agent.epsilon = 0.0

    # Run test episodes.
    while True:
        obs, info = env.reset()
        done = truncated = False
        total_reward = 0.0
        while not (done or truncated):
            action = agent.select_action(obs)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            env.render()
        print("Episode finished, total reward:", total_reward)

if __name__ == "__main__":
    test_model()
