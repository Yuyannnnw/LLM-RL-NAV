import gymnasium as gym
import highway_env
from stable_baselines3 import DQN

def call_llm_for_shaping(prev_obs, next_obs, action):
    """
    Placeholder that returns a numeric shaping value.
    In a real scenario:
      1) Build a prompt from (prev_obs, next_obs, action).
      2) Query the LLM.
      3) Parse the output into a float reward signal.

    Here we just return 0.0 as a dummy placeholder.
    """
    return 0.0

class LLMShapingWrapper(gym.Wrapper):
    """
    A wrapper that:
      1. Keeps the environment's base reward as-is.
      2. Adds an LLM-based shaping term each step, using both
         the old (prev_obs) and new (next_obs) state if desired.
    """
    def __init__(self, env):
        super().__init__(env)
        self.prev_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_obs = obs
        return obs, info

    def step(self, action):
        # Standard call to the underlying env
        next_obs, base_reward, done, truncated, info = self.env.step(action)

        # Optionally call an LLM-based function with (prev_obs, next_obs, action)
        shaping_term = call_llm_for_shaping(self.prev_obs, next_obs, action)

        # Combine the environment's base reward with the shaping term
        total_reward = base_reward + shaping_term

        # Update stored observation
        self.prev_obs = next_obs

        return next_obs, total_reward, done, truncated, info


def main():
    # 1) Make the base highway-fast environment
    env = gym.make("highway-fast-v0")

    # 2) Wrap it with our LLMShapingWrapper (which keeps base reward)
    env = LLMShapingWrapper(env)

    # 3) Build a DQN model from stable-baselines3
    model = DQN('MlpPolicy', env,
              policy_kwargs=dict(net_arch=[256, 256]),
              learning_rate=5e-4,
              buffer_size=15000,
              learning_starts=200,
              batch_size=32,
              gamma=0.8,
              train_freq=1,
              gradient_steps=1,
              target_update_interval=50,
              verbose=1,
              tensorboard_log="highway_dqn/")

    # 4) Train the model for 20,000 timesteps
    model.learn(total_timesteps=2e4)

    # 5) Save the trained model
    model.save("highway_dqn_sb3/model")
    print("Training complete and model saved.")


if __name__ == "__main__":
    main()
