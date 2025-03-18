import os
import csv
import matplotlib.pyplot as plt
import json

def load_env_config(config_path = "configurations/env_config.json"):
    """
    Loads the JSON config file for environment settings.
    """
    if not os.path.isfile(config_path):
        print(f"No config file found at {config_path}, proceeding with defaults.")
        return {}
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    return config_data

def build_model_path(folder_name, env_id, mode):
    """
    Create a cross-platform path for the model file in nested folders:
    folder_name/env_id/mode/model

    Replaces certain special characters in env_id or mode if needed.
    """
    # Clean up strings to avoid issues with special chars:
    env_id_clean = env_id.replace('-', '_').replace(' ', '_')
    mode_clean = mode.replace(' ', '_')

    # Build nested directory path:
    nested_dir = os.path.join(folder_name, env_id_clean, mode_clean)
    
    # Final filename for the model
    filename = "model" 

    return os.path.join(nested_dir, filename)

def build_logs_path(folder_name, env_id, mode):
    """
    Returns a nested folder path for logs:
      folder_name/env_id/mode/
    e.g. monitor_logs/highway_fast_v0/Hybrid/
    """
    env_id_clean = env_id.replace('-', '_').replace(' ', '_')
    mode_clean = mode.replace(' ', '_')
    return os.path.join(folder_name, env_id_clean, mode_clean)


def get_user_choices():
    """
    Prompt user for environment and training type, 
    return (env_id, mode) tuple.
    """
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

    print("Choose training type:")
    print("1. RL (standard base reward)")
    print("2. Hybrid (LLM + collision penalty)")
    train_choice = input("Enter the number of your choice: ")

    if train_choice == '1':
        mode = 'RL'
    elif train_choice == '2':
        mode = 'Hybrid'
    else:
        print("Invalid choice, defaulting to RL.")
        mode = 'RL'

    return env_id, mode

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