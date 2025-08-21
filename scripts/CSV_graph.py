import os
from datetime import datetime

# Get project root directory (assuming this script is in the project root or you need to adjust the path)
def get_project_root():
    return os.path.dirname(os.path.abspath(__file__))

# Prepare reward history
reward_history = df.iloc[:, 0].values  # extract the single column

# Define early and late segments (first and last 10%)
segment_size = int(len(reward_history) * 0.1)
early = reward_history[:segment_size]
late = reward_history[-segment_size:]

# Perform t-test
t_stat, p_value = ttest_ind(early, late)

# Dummy parameters text (as CSV doesn't contain it)
params_text = (
    "Seed: UNKNOWN | Learning Rate: UNKNOWN | Episodes: {}\n"
    "Agents: ? | Gamma: ? | Epsilon Start: ?\n"
    "Epsilon Decay: ? | Epsilon Min: ? | Max Steps: ?"
).format(len(reward_history))


# Define the visualization function for PDF output
def visualize_results(reward_history, early, late, t_stat, p_value, filename='recreated_plot.pdf', params_text=''):
    plt.figure(figsize=(12, 8))
    gs = plt.GridSpec(2, 1, height_ratios=[1, 4])

    ax1 = plt.subplot(gs[0])
    ax1.text(0.05, 0.5, params_text, fontsize=9, fontfamily='monospace', verticalalignment='center')
    ax1.axis('off')

    ax2 = plt.subplot(gs[1])
    '''ax2.plot(reward_history, alpha=0.2, color='blue', label='Raw rewards')'''

    window_short = 100
    window_long = 500
    smoothed_short = np.convolve(reward_history, np.ones(window_short) / window_short, mode='valid')
    smoothed_long = np.convolve(reward_history, np.ones(window_long) / window_long, mode='valid')

    ax2.plot(range(window_short - 1, len(smoothed_short) + window_short - 1),
             smoothed_short, color='red', linewidth=2, label=f'{window_short}-episode MA')
    ax2.plot(range(window_long - 1, len(smoothed_long) + window_long - 1),
             smoothed_long, color='green', linewidth=2, label=f'{window_long}-episode MA')

    ax2.axhline(y=early.mean(), color='orange', linestyle='--', alpha=0.5, label=f'Early mean: {early.mean():.2f}')
    ax2.axhline(y=late.mean(), color='purple', linestyle='--', alpha=0.5, label=f'Late mean: {late.mean():.2f}')

    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Reward')
    ax2.set_title(f'Training Progress\nImprovement: {((late.mean() - early.mean()) / abs(early.mean()) * 100):.2f}%')

    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.text(0.02, 0.02, f'p-value: {p_value:.2e}\nt-stat: {t_stat:.2f}',
             transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', format='pdf')
    print(f"Plot saved as {filename}")
    plt.show()
    plt.close()


# Create the regenerated graphs directory structure
project_root = get_project_root()
regenerated_graphs_dir = os.path.join(project_root, "plots", "regenerated_graphs")
os.makedirs(regenerated_graphs_dir, exist_ok=True)

# Generate timestamp for unique filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = os.path.join(regenerated_graphs_dir, f"recreated_plot_{timestamp}.pdf")

# Generate and show the plot with PDF output in the correct directory
visualize_results(reward_history, early, late, t_stat, p_value, filename=filename,
                  params_text=params_text)