import os
import csv
import random
import numpy as np
import gymnasium as gym
import highway_env
from tqdm import trange

from stable_baselines3 import DQN

# Utility functions
from utils.utils import get_user_choices, build_model_path, build_logs_path, load_env_config
def main():
    # 1) Prompt user for configuration settings
    env_id, mode, observation_type, run_id = get_user_choices()
    env_config = load_env_config(observation_type)

    # 2) Prompt user for how many episodes to evaluate
    num_test_episodes = input("How many test episodes do you want to run? ")
    try:
        num_test_episodes = int(num_test_episodes)
    except ValueError:
        num_test_episodes = 100  # fallback

    # 3) Prompt user for a seed (optional)
    seed_input = input("Enter a random seed (integer) for reproducibility (press enter to skip): ")
    try:
        seed = int(seed_input)
    except ValueError:
        seed = None

    # 4) Build the path to the model
    folder_name = "models"
    model_path = build_model_path(folder_name, env_id, mode, observation_type, run_id)
    model_file = model_path + ".zip"

    # 5) Check if the .zip file exists
    if not os.path.isfile(model_file):
        print(f"Model not found at: {model_file}")
        print("Make sure you trained it first or that your paths match.")
        return

    # 6) Create the environment
    base_env = gym.make(env_id, render_mode=None, config=env_config)

    # 7) Handle random seed
    if seed is not None:
        print(f"Using random seed = {seed}")
        random.seed(seed)
        np.random.seed(seed)
        base_env.reset(seed=seed)
    else:
        print("No seed specified; results may vary run to run.")

    # 8) Load the model
    print(f"Loading model from: {model_file}")
    model = DQN.load(model_path, env=base_env)

    # 9) Prepare for evaluation logs
    eval_logs_folder = "eval_logs"
    eval_logs_dir = build_logs_path(eval_logs_folder, env_id, mode, observation_type, run_id)
    os.makedirs(eval_logs_dir, exist_ok=True)
    csv_path = os.path.join(eval_logs_dir, "eval.csv")

    # 10) Run test episodes with a progress bar
    results = []  # Will store (episode_index, reward, success_int)
    print(f"Evaluating for {num_test_episodes} episodes...")

    for episode in trange(num_test_episodes, desc="Evaluation Progress"):
        obs, info = base_env.reset()
        done, truncated = False, False
        episode_reward = 0.0
        collision_occurred = False

        while not (done or truncated):
            # Predict next action (deterministic)
            action, _= model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = base_env.step(action)
            episode_reward += reward

            # Check collision
            if info.get("crashed", False):
                collision_occurred = True

        # Record success as 0 or 1
        success_int = 0 if collision_occurred else 1
        results.append((episode + 1, episode_reward, success_int))

    # 11) Once all episodes are done, compute summary stats
    print("\nEvaluation complete.")

    num_episodes = len(results)
    total_reward = sum(r for (_, r, _) in results)
    average_reward = total_reward / num_episodes if num_episodes > 0 else 0.0

    # success_int is 1 if no collision, 0 if collision
    total_successes = sum(s for (_, _, s) in results)
    success_rate = (total_successes / num_episodes * 100.0) if num_episodes > 0 else 0.0

    print(f"  Number of episodes: {num_episodes}")
    print(f"  Average reward: {average_reward:.2f}")
    print(f"  Success rate (no collision): {success_rate:.1f}%")

    # 12) Write results to CSV
    print("Saving evaluation logs...")
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "success"])  # success is 0 or 1
        for (ep_idx, rew, success) in results:
            writer.writerow([ep_idx, rew, success])

    print(f"Evaluation logs saved to: {csv_path}")


if __name__ == "__main__":
    main()
