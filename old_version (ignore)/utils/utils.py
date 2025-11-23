import os
import csv
import matplotlib.pyplot as plt
import json

def load_env_config(observation_type, base_path="configurations/env_config"):
    """
    Loads the JSON config file for environment settings based on observation_type.
    Looks for:
      <base_path>_<observation_type>.json
    e.g. configurations/env_config_full.json or configurations/env_config_partial.json
    """
    # Build the config filename
    obs_clean = observation_type.replace('-', '_').replace(' ', '_')
    config_path = f"{base_path}_{obs_clean}.json"

    if not os.path.isfile(config_path):
        print(f"No config file found at {config_path}, proceeding with defaults.")
        return {}

    with open(config_path, 'r') as f:
        config_data = json.load(f)
    return config_data

def build_model_path(folder_name, env_id, mode, observation_type, run_id=None):
    """
    Create a cross-platform path for the model file in nested folders:
      folder_name/env_id/mode/observation_type/model[_run_id].zip

    If run_id is provided, it will be appended to the filename only.
    """
    # Clean up inputs
    env_clean  = clean(env_id)
    mode_clean = clean(mode)
    obs_clean  = clean(observation_type)

    # Build the directory path
    nested_dir = os.path.join(folder_name, env_clean, mode_clean, obs_clean)
    os.makedirs(nested_dir, exist_ok=True)

    # Build the filename
    if run_id is not None:
        run_clean = clean(run_id)
        filename = f"model_{run_clean}"
    else:
        filename = "model"

    # SB3 will append .zip automatically
    return os.path.join(nested_dir, filename)


def build_logs_path(folder_name, env_id, mode, observation_type, run_id):
    """
    Returns a nested folder path under 'logs/':
      logs/<folder_name>/<env_id>/<mode>/<observation_type>/<run_id>/

    All inputs are sanitized (spaces & hyphens → underscores).
    """
    # Clean up inputs
    env_clean  = clean(env_id)
    mode_clean = clean(mode)
    obs_clean  = clean(observation_type)
    run_clean  = clean(run_id)

    return os.path.join(
        "logs",
        folder_name,
        env_clean,
        mode_clean,
        obs_clean,
        run_clean
    )

def get_user_choices():
    """
    Prompt user for environment, training type, and observation mode.
    Returns a tuple: (env_id, mode, observation)
    """
    # Environment selection
    print("Choose environment:")
    print("1. highway-fast-v0")
    print("2. highway-v0")
    print("3. merge-v0")
    env_choice = input("Enter the number of your choice: ")

    if env_choice == '1':
        env_id = "highway-fast-v0"
    elif env_choice == '2':
        env_id = "highway-v0"
    elif env_choice == '3':
        env_id = "merge-v0"
    else:
        print("Invalid choice, defaulting to 'highway-fast-v0'")
        env_id = "highway-fast-v0"

    # Training type selection
    print("\nChoose training type:")
    print("1. RL (standard base reward)")
    print("2. Hybrid (LLM + collision penalty)")
    train_choice = input("Enter the number of your choice: ")

    if train_choice == '1':
        mode = 'RL'
    elif train_choice == '2':
        mode = 'Hybrid'
    else:
        print("Invalid choice, defaulting to 'RL'")
        mode = 'RL'

    # Observation mode selection
    print("\nChoose observation mode:")
    print("1. Full observation")
    print("2. Partial observation")
    obs_choice = input("Enter the number of your choice: ")

    if obs_choice == '1':
        observation = 'full'
    elif obs_choice == '2':
        observation = 'partial'
    else:
        print("Invalid choice, defaulting to 'full'")
        observation = 'full'
    
    # Optional run ID
    print("\nEnter a run ID to save this model under (or press Enter to overwrite existing model):")
    run_id_input = input("Run ID: ").strip()
    run_id = run_id_input if run_id_input else None

    return env_id, mode, observation, run_id

def parse_monitor_logs(monitor_csv_path):
    """
    Reads a CSV with columns assumed to be [r, l, t].
    Skips the first two lines (#comment and header).
    Returns:
      - row_indices: list of integers (1..N) for each row
      - rewards: list of floats from column 0 (r)
      - times:   list of floats from column 2 (t)
    """
    row_indices = []
    rewards = []
    times = []

    if not os.path.isfile(monitor_csv_path):
        print(f"Monitor CSV not found: {monitor_csv_path}")
        return row_indices, rewards, times

    with open(monitor_csv_path, "r") as f:
        reader = csv.reader(f)
        # Skip the first two lines: comment + header
        next(reader, None)
        next(reader, None)

        row_count = 0
        for row in reader:
            # row expected: [r_str, l_str, t_str]
            if len(row) < 3:
                continue
            row_count += 1

            r_val = float(row[0])  # reward
            t_val = float(row[2])  # time

            row_indices.append(row_count)
            rewards.append(r_val)
            times.append(t_val)

    return row_indices, rewards, times

def parse_eval_logs(eval_csv_path):
    """
    Reads an evaluation CSV with columns: episode, reward, success
    Returns:
      episodes_eval -> list of episodes
      rewards_eval -> list of rewards
      successes_eval -> list of 0/1 for success
    """
    episodes_eval = []
    rewards_eval = []
    successes_eval = []

    if not os.path.isfile(eval_csv_path):
        print(f"Evaluation CSV not found: {eval_csv_path}")
        return episodes_eval, rewards_eval, successes_eval

    with open(eval_csv_path, "r") as f:
        reader = csv.reader(f)
        # Assume first line is header
        header = next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            ep = int(row[0])
            rew = float(row[1])
            suc = int(row[2])  # 0 or 1
            episodes_eval.append(ep)
            rewards_eval.append(rew)
            successes_eval.append(suc)

    return episodes_eval, rewards_eval, successes_eval

def plot_time_series(x_values, y_values, x_label, y_label, title):
    """
    A generic time-series plot:
      x_values -> array-like for X-axis
      y_values -> array-like for Y-axis
      x_label, y_label -> strings for axis labels
      title -> figure title
    """
    plt.figure()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x_values, y_values)
    plt.grid(True)

def plot_evaluation_summary_table(total_episodes, total_collisions, success_rate):
    """
    Creates a small Matplotlib figure with a table
    summarizing evaluation statistics.
    """
    fig, ax = plt.subplots()
    ax.set_axis_off()  # hide the main axes

    # Prepare table data
    table_data = [
        ["Total episodes", str(total_episodes)],
        ["Total collisions", str(total_collisions)],
        [ "Success rate", f"{success_rate:.1f}%" ]
    ]

    # Create the table in the center
    table = ax.table(cellText=table_data,
                     colLabels=None,  # no column headers
                     loc='center')
    table.scale(1, 2)  # optionally adjust table size
    ax.set_title("Evaluation Summary", fontsize=12, pad=10)

def clean(s):
    return str(s).replace('-', '_').replace(' ', '_')