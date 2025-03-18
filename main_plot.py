import os
import matplotlib.pyplot as plt

# Import your helper functions from utils
from utils.utils import (
    get_user_choices,
    build_logs_path,
    parse_monitor_logs,
    parse_eval_logs,
    plot_time_series,
    plot_evaluation_summary_table,
)

def main():
    # 1) Prompt user for environment + mode
    env_id, mode = get_user_choices()

    # Create a directory for saving plots: plots/<env_id>/<mode>
    plot_dir = build_logs_path("plots", env_id, mode)
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Plots will be saved to: {plot_dir}\n")

    # ------------------------------------------------------------------
    # PART A: TRAINING (Monitor logs) -> [r, l, t]
    # ------------------------------------------------------------------
    monitor_base = "monitor_logs"
    monitor_dir = build_logs_path(monitor_base, env_id, mode)
    monitor_csv_path = os.path.join(monitor_dir, "monitor.csv")

    row_indices, rewards, times = parse_monitor_logs(monitor_csv_path)
    if row_indices and rewards:
        # Figure 1: training reward
        title_r = f"Training Reward\n({env_id}, {mode})"
        plt.figure()
        plot_time_series(
            x_values=row_indices,
            y_values=rewards,
            x_label="Row Index",
            y_label="Reward (r)",
            title=title_r
        )
        reward_fig_path = os.path.join(plot_dir, "training_reward.png")
        plt.savefig(reward_fig_path)
        plt.close()
        print(f"Saved training reward plot to {reward_fig_path}")

    if row_indices and times:
        # Figure 2: training time
        title_t = f"Training Time\n({env_id}, {mode})"
        plt.figure()
        plot_time_series(
            x_values=row_indices,
            y_values=times,
            x_label="Row Index",
            y_label="Time (t)",
            title=title_t
        )
        time_fig_path = os.path.join(plot_dir, "training_time.png")
        plt.savefig(time_fig_path)
        plt.close()
        print(f"Saved training time plot to {time_fig_path}")

    # ------------------------------------------------------------------
    # PART B: EVALUATION (episode, reward, success)
    # ------------------------------------------------------------------
    eval_base = "eval_logs"
    eval_dir = build_logs_path(eval_base, env_id, mode)
    eval_csv_path = os.path.join(eval_dir, "eval.csv")

    episodes_eval, rewards_eval, successes_eval = parse_eval_logs(eval_csv_path)
    if episodes_eval and rewards_eval:
        # Figure 3: evaluation reward
        title_eval = f"Evaluation Reward per Episode\n({env_id}, {mode})"
        plt.figure()
        plot_time_series(
            x_values=episodes_eval,
            y_values=rewards_eval,
            x_label="Episode",
            y_label="Reward",
            title=title_eval
        )
        eval_fig_path = os.path.join(plot_dir, "evaluation_reward.png")
        plt.savefig(eval_fig_path)
        plt.close()
        print(f"Saved evaluation reward plot to {eval_fig_path}")

        # Summarize collisions & success
        total_episodes = len(episodes_eval)
        total_successes = sum(successes_eval)
        total_collisions = total_episodes - total_successes
        success_rate = (total_successes / total_episodes * 100.0) if total_episodes else 0.0

        # Create a separate figure for the evaluation summary table
        plt.figure()
        plot_evaluation_summary_table(total_episodes, total_collisions, success_rate)
        summary_fig_path = os.path.join(plot_dir, "evaluation_summary.png")
        plt.savefig(summary_fig_path)
        plt.close()
        print(f"Saved evaluation summary table to {summary_fig_path}")

    else:
        print("No evaluation data found or file not found.\nSkipping eval reward plot & summary table.")

    print("\nAll requested plots have been saved successfully.")

if __name__ == "__main__":
    main()
