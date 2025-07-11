import csv
import os
import numpy as np
from scipy import stats
from datetime import datetime

'''
1. Run training only: python main.py --train=true --evaluate=false --force_retrain=true
2. Run evaluation only: python main.py --train=false --evaluate=true --force_retrain=false
3. THIS IS THE MAIN ONE -> Run both: python main.py --train=true --evaluate=true --force_retrain=true
'''
import matplotlib.pyplot as plt
from absl import app
from absl import flags

from common.pettingzoo_environment import SimpleSpreadEnv
from runner import RunnerSimpleSpreadEnv
from utils.config_utils import ConfigObjectFactory

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', True, 'Run training')
flags.DEFINE_boolean('evaluate', False, 'Run evaluation')
flags.DEFINE_boolean('force_retrain', False, 'Force retraining even if saved model exists')

def main(argv):
    env = SimpleSpreadEnv()
    try:
        if FLAGS.train:
            print("Starting training...")
            runner = RunnerSimpleSpreadEnv(env)
            if FLAGS.force_retrain:
                # Clear existing results
                runner.agents.del_model()
                if os.path.exists(runner.result_path):
                    os.remove(runner.result_path)
                if os.path.exists(runner.memory_path):
                    os.remove(runner.memory_path)
                    
            runner.run_marl()
            print("Training completed")
            
        if FLAGS.evaluate:
            print("Starting evaluation...")
            evaluate()
            print("Evaluation completed")
            
    finally:
        env.close()



def evaluate():
    try:
        train_config = ConfigObjectFactory.get_train_config()
        env_config = ConfigObjectFactory.get_environment_config()
        csv_filename = os.path.join(train_config.result_dir, env_config.learn_policy, "result.csv")
        
        # Check if results file exists
        if not os.path.exists(csv_filename):
            print(f"Error: Results file not found at {csv_filename}")
            return
            
        # Read all rewards from CSV
        total_rewards = []
        with open(csv_filename, 'r') as f:
            r_csv = csv.reader(f)
            for data in r_csv:
                total_rewards.append(float(data[0]))
        
        if not total_rewards:
            print("Error: No reward data found in results file")
            return
            
        print(f"Processing {len(total_rewards)} reward entries")
        
        # Rest of the visualization code...
        # (Keep the visualization code from my previous response here)
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
    train_config = ConfigObjectFactory.get_train_config()
    env_config = ConfigObjectFactory.get_environment_config()
    csv_filename = os.path.join(train_config.result_dir, env_config.learn_policy, "result.csv")

    # Read all rewards from CSV
    total_rewards = []
    with open(csv_filename, 'r') as f:
        r_csv = csv.reader(f)
        for data in r_csv:
            total_rewards.append(float(data[0]))

    rewards = np.array(total_rewards)

    # Calculate statistics for analysis
    analysis_window = min(1000, len(rewards) // 10)  # Use 10% of data or 1000 points
    early = rewards[:analysis_window]
    late = rewards[-analysis_window:]

    # Calculate statistical significance
    t_stat, p_value = stats.ttest_ind(early, late)

    # Create visualization
    plt.figure(figsize=(12, 8))
    gs = plt.GridSpec(2, 1, height_ratios=[1, 4])

    # Parameters panel
    ax1 = plt.subplot(gs[0])
    params_text = (
        f"Parameters:\n"
        f"Policy: {env_config.learn_policy} | Grid Size: {env_config.grid_size} | "
        f"Max Cycles: {env_config.max_cycles}\n"
        f"Epochs: {train_config.epochs} | "
        f"Evaluate Epoch: {train_config.evaluate_epoch} | "
        f"CUDA: {train_config.cuda}"
    )
    ax1.text(0.05, 0.5, params_text, fontsize=9, fontfamily='monospace',
             verticalalignment='center')
    ax1.axis('off')

    # Main plot
    ax2 = plt.subplot(gs[1])
    ax2.plot(rewards, alpha=0.2, color='blue', label='Raw rewards')

    # Add moving averages
    window_short = 100
    window_long = 500
    if len(rewards) > window_long:
        smoothed_short = np.convolve(rewards, np.ones(window_short) / window_short, mode='valid')
        smoothed_long = np.convolve(rewards, np.ones(window_long) / window_long, mode='valid')

        ax2.plot(range(window_short - 1, len(smoothed_short) + window_short - 1),
                 smoothed_short, color='red', linewidth=2,
                 label=f'{window_short}-episode MA')
        ax2.plot(range(window_long - 1, len(smoothed_long) + window_long - 1),
                 smoothed_long, color='green', linewidth=2,
                 label=f'{window_long}-episode MA')

    # Add baseline means
    ax2.axhline(y=early.mean(), color='orange', linestyle='--', alpha=0.5,
                label=f'Early mean: {early.mean():.2f}')
    ax2.axhline(y=late.mean(), color='purple', linestyle='--', alpha=0.5,
                label=f'Late mean: {late.mean():.2f}')

    # Add labels and title
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Reward')
    improvement = ((late.mean() - early.mean()) / abs(early.mean()) * 100)
    ax2.set_title(f'Training Progress\nImprovement: {improvement:.2f}%')

    # Add statistical information
    ax2.text(0.02, 0.02, f'p-value: {p_value:.2e}\nt-stat: {t_stat:.2f}',
             transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.8))

    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(train_config.result_dir, env_config.learn_policy,
                            f"training_visualization_{timestamp}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {filename}")
    plt.show()
    plt.close()


if __name__ == "__main__":
    app.run(main)