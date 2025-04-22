import gymnasium as gym
import highway_env
from stable_baselines3 import DQN
from utils.utils import get_user_choices, load_env_config, build_model_path

env_id, mode, observation_type, run_id = get_user_choices()
env_config = load_env_config(observation_type)
env = gym.make(env_id, render_mode="rgb_array", config=env_config)
folder_name = "models"
model_save_path = build_model_path(folder_name, env_id, mode, observation_type, run_id)

model = DQN.load(model_save_path)
while True:
  done = truncated = False
  obs, info = env.reset()
  while not (done or truncated):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()