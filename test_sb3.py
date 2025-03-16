from stable_baselines3 import DQN
import gymnasium
import highway_env
import numpy as np

from DQN.Agent import DQNAgent  # Assuming your DQNAgent class is in Agent.py

def test_model():
    # Create the environment.
    env = gymnasium.make("highway-fast-v0", render_mode='rgb_array')

    # Initialize the agent with the same hyperparameters used during training.
    model = DQN.load("highway_dqn_sb3/model")
    

    # Run test episodes.
    while True:
        total_reward = 0.0
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            env.render()
        print("Episode finished, total reward:", total_reward)

if __name__ == "__main__":
    test_model()
