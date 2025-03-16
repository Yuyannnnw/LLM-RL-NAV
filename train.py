import gymnasium
import highway_env
import numpy as np
from DQN.Agent import DQNAgent 
from DQN.Logger import TrainingLogger

def main():
    # Create the environment.
    env = gymnasium.make("highway-fast-v0")
    # Compute the flattened state dimension.
    state_dim = int(np.prod(env.observation_space.shape))
    action_dim = env.action_space.n

    # Create a logger (you can later extend this to log TensorBoard data if desired).
    logger = TrainingLogger(log_dir="highway_dqn", save_freq=50)

    # Create the agent with hyperparameters matching the SB3 example.
    agent = DQNAgent(state_dim, action_dim, hidden_layers=[256, 256],
                      lr=5e-4,
                      gamma=0.8,
                      epsilon=1.0,
                      epsilon_decay=0.995,
                      min_epsilon=0.01,
                      target_update_freq=50,
                      buffer_capacity=15000)

    # Train the agent.
    agent.learn(env, num_episodes=100, batch_size=32, max_steps=2000,
                learning_starts=200, train_freq=1, gradient_steps=1, verbose=1, logger=logger)

    # Save the trained model.
    agent.save("highway_dqn/model.pth")

if __name__ == "__main__":
    main()