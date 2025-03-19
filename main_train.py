import os
import gymnasium as gym
import ollama
import highway_env
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from tqdm import trange  # for the progress bar

# Utility functions
from utils.utils import get_user_choices, build_model_path, build_logs_path, load_env_config

def call_llm_for_shaping(prev_obs, next_obs, action):
    """
    Queries an LLM to adjust the reward based on previous state, next state, action, and base reward.
    """

    prompt = f"""
    You are a reinforcement learning assistant helping to fine-tune rewards of an autonomous vehicle.
    Only response a numerical float. Do not give me any other information.
    The value should be in range 0 to 10, where higher value refers to encourage human-like action.
    Action space is discrete where 0: 'LANE_LEFT', 1: 'IDLE', 2: 'LANE_RIGHT', 3: 'FASTER', 4: 'SLOWER'.   
    Given the following information:
    - Previous Observation: {prev_obs}
    - Action Taken: {action}
    - New Observation: {next_obs}
    
    Adjust the reward to improve learning.  
    """
    #print("Prompt to LLM:")
    response = ollama.chat(
        model='llama3.2',
        messages=[{'role': 'user', 'content': prompt}]
    )
    #print(f"LLM Response: {response}")
    
    try:
        # Extract numerical adjustment from LLM response
        shaping_value = float(response['message']['content'].strip())
    except ValueError:
        shaping_value = 0.0  # Default to 0 if parsing fails

    #print(f"Shaping Value: {shaping_value}")
    
    return shaping_value

class EnvWrapper(gym.Wrapper):
    """
    Applies either the base environment reward (RL mode)
    or collision penalty + LLM shaping (Hybrid mode).
    """
    def __init__(self, env, mode='RL'):
        super().__init__(env)
        self.mode = mode
        self.prev_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_obs = obs
        return obs, info

    def step(self, action):
        next_obs, base_reward, done, truncated, info = self.env.step(action)

        if self.mode == 'RL':
            total_reward = base_reward
        elif self.mode == 'Hybrid':
            collision_penalty = -1.0 if info.get("crashed", False) else 0.0
            shaping_term = call_llm_for_shaping(self.prev_obs, next_obs, action) / 10.0
            total_reward = collision_penalty + shaping_term
        else:
            total_reward = base_reward

        self.prev_obs = next_obs
        return next_obs, total_reward, done, truncated, info

def main():
    # 1) Prompt user for environment and mode
    env_id, mode = get_user_choices()
    env_config = load_env_config()

    # 2) Construct path for logs: monitor_logs/env_id/mode/
    logs_base = "monitor_logs"
    monitor_dir = build_logs_path(logs_base, env_id, mode)
    os.makedirs(monitor_dir, exist_ok=True)
    monitor_csv_path = os.path.join(monitor_dir, "monitor.csv")

    # 3) Create base environment
    base_env = gym.make(env_id, render_mode=None, config=env_config)

    # 4) Wrap it with Monitor to log to CSV
    monitored_env = Monitor(base_env, filename=monitor_csv_path)

    # 5) Then wrap with EnvWrapper so the Monitor sees final shaped reward
    env = EnvWrapper(monitored_env, mode=mode)

    # 6) Build DQN
    model = DQN(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=5e-4,
        buffer_size=15000,
        learning_starts=200,
        batch_size=32,
        gamma=0.8,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=50,
        verbose=0,
    )
    total_timesteps = 20_000

    # 7) Chunked training with a progress bar after each chunk
    chunks_num = 10
    chunk_size = total_timesteps // chunks_num
    chunks = total_timesteps // chunk_size  

    print(f"Training {mode} on {env_id} for {total_timesteps} timesteps in {chunks} chunks...")
    timesteps_so_far = 0
    for i in trange(chunks, desc="Training progress"):
        model.learn(total_timesteps=chunk_size, reset_num_timesteps=False)
        timesteps_so_far += chunk_size

    # If there's leftover timesteps (if total_timesteps wasn't divisible by chunk_size),
    # handle them as well. E.g.
    remainder = total_timesteps - timesteps_so_far
    if remainder > 0:
        model.learn(total_timesteps=remainder, reset_num_timesteps=False)

    # 8) Build path for model: models/env_id/mode/model
    folder_name = "models"
    model_save_path = build_model_path(folder_name, env_id, mode)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # 9) Save the model
    model.save(model_save_path)
    print(f"Model saved to: {model_save_path}")

    # 10) Print location of monitor logs
    print(f"Monitor logs saved to: {monitor_csv_path}")


if __name__ == "__main__":
    main()
