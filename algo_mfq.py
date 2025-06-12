import numpy as np
from collections import defaultdict
from pettingzoo.mpe.simple_spread_v3 import parallel_env
import supersuit as ss
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime
import matplotlib
import pickle  # Add this line
matplotlib.use('TkAgg')  # or 'Qt5Agg' depending on your system

# --- Hyperparameters ---
N_EPISODES = 10000
MAX_STEPS = 25
LEARNING_RATE = 0.01
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.9997
N_AGENTS = 3
ANALYSIS_WINDOW = 500


class MFQAgent:
    def __init__(
        self,
        obs_space,
        action_space,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        epsilon=EPSILON_START,
        epsilon_decay=EPSILON_DECAY,
        min_epsilon=EPSILON_MIN,
    ):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.action_space = action_space
        # Initialize as regular dictionary instead of defaultdict
        self.q_table = {}
        self.visit_counts = {}
        
    def get_state_key(self, obs):
        state = tuple(np.round(obs, 1))
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space.n)
            self.visit_counts[state] = np.zeros(self.action_space.n)
        return state

    def act(self, obs):
        state = self.get_state_key(obs)
        
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()
        return int(np.argmax(self.q_table[state]))

    def learn(self, state, action, reward, next_state, mean_action):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        # Mean field Q-learning update
        avg_next_q = np.mean(self.q_table[next_state_key])
        td_target = reward + self.gamma * (0.9 * avg_next_q + 0.1 * mean_action)
        td_error = td_target - self.q_table[state_key][action]
        
        # Adaptive learning rate based on visit count
        self.visit_counts[state_key][action] += 1
        visits = self.visit_counts[state_key][action]
        alpha = self.lr / (1 + 0.1 * visits)
        
        self.q_table[state_key][action] += alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

# Environment setup
env = parallel_env(N=N_AGENTS, local_ratio=0.7, max_cycles=MAX_STEPS)
env = ss.dtype_v0(env, dtype="float32")
env = ss.clip_reward_v0(env)

# Initialize agents
agents = {
    name: MFQAgent(env.observation_space(name), env.action_space(name))
    for name in env.possible_agents
}

# Training loop
reward_history = []
episode_lengths = []

for episode in range(N_EPISODES):
    if episode % 100 == 0:
        print(f"Starting episode {episode}")
    
    obs, _ = env.reset(seed=episode)
    dones = {agent: False for agent in env.possible_agents}
    total_reward = 0
    step_count = 0
    
    while not all(dones.values()) and step_count < MAX_STEPS:
        step_count += 1
        active_agents = env.agents
        
        if not active_agents:
            break
            
        # Get actions for all agents
        actions = {
            agent: agents[agent].act(obs[agent])
            for agent in active_agents if not dones[agent]
        }
        
        # Calculate mean action for active agents
        if actions:  # Only if there are active agents
            mean_action = np.mean(list(actions.values()))
        else:
            mean_action = 0
        
        # Environment step
        next_obs, rewards, terminations, truncations, _ = env.step(actions)
        total_reward += sum(rewards.values())
        
        # Update all agents
        for agent in active_agents:
            if not dones[agent]:
                agents[agent].learn(
                    obs[agent],
                    actions[agent],
                    rewards[agent],
                    next_obs[agent],
                    mean_action
                )
        
        # Update observations and done states
        obs = next_obs
        for agent in env.possible_agents:
            dones[agent] = terminations.get(agent, False) or truncations.get(agent, False)
    
    # Record rewards and decay epsilon
    reward_history.append(total_reward)
    episode_lengths.append(step_count)
    
    # Decay epsilon for all agents
    for agent in agents.values():
        agent.decay_epsilon()
        
    if episode % 100 == 0:
        print(f"Episode {episode} completed in {step_count} steps with total reward: {total_reward:.2f}")

# Analysis
early = np.array(reward_history[:ANALYSIS_WINDOW])
late = np.array(reward_history[-ANALYSIS_WINDOW:])

print("\nPerformance Analysis:")
print(f"Early mean±std: {early.mean():.2f} ± {early.std():.2f}")
print(f"Late mean±std: {late.mean():.2f} ± {late.std():.2f}")
print(f"Improvement: {((late.mean() - early.mean()) / abs(early.mean()) * 100):.2f}%")

t_stat, p_value = stats.ttest_ind(early, late)
print(f"Statistical significance:")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.6f}")

def visualize_results(reward_history, early, late, t_stat, p_value):
    fig = plt.figure(figsize=(12, 8))
    
    def handle_close(evt):
        plt.close('all')
    
    fig.canvas.mpl_connect('close_event', handle_close)
    
    gs = plt.GridSpec(2, 1, height_ratios=[1, 4])
    
    # Parameters subplot
    ax1 = plt.subplot(gs[0])
    param_text = (
        f"Parameters:\n"
        f"Learning Rate: {LEARNING_RATE}  |  Episodes: {N_EPISODES}  |  Agents: {N_AGENTS}  |  Gamma: {GAMMA}\n"
        f"Epsilon Start: {EPSILON_START}  |  Epsilon Decay: {EPSILON_DECAY}  |  Epsilon Min: {EPSILON_MIN}  |  Max Steps: {MAX_STEPS}"
    )
    ax1.text(0.05, 0.5, param_text, fontsize=9, fontfamily='monospace', verticalalignment='center')
    ax1.axis('off')
    
    # Main plot
    ax2 = plt.subplot(gs[1])
    ax2.plot(reward_history, alpha=0.2, color='blue', label='Raw rewards')
    
    window_short = 100
    window_long = 500
    smoothed_short = np.convolve(reward_history, np.ones(window_short)/window_short, mode='valid')
    smoothed_long = np.convolve(reward_history, np.ones(window_long)/window_long, mode='valid')
    
    ax2.plot(range(window_short-1, len(smoothed_short)+window_short-1), 
             smoothed_short, 
             color='red', 
             linewidth=2, 
             label=f'{window_short}-episode MA')
    
    ax2.plot(range(window_long-1, len(smoothed_long)+window_long-1), 
             smoothed_long, 
             color='green', 
             linewidth=2, 
             label=f'{window_long}-episode MA')
    
    ax2.axhline(y=early.mean(), color='orange', linestyle='--', alpha=0.5, 
               label=f'Early mean: {early.mean():.2f}')
    ax2.axhline(y=late.mean(), color='purple', linestyle='--', alpha=0.5,
               label=f'Late mean: {late.mean():.2f}')
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Reward')
    ax2.set_title(f'Training Progress over {N_EPISODES} Episodes\nImprovement: {((late.mean() - early.mean()) / abs(early.mean()) * 100):.2f}%')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax2.text(0.02, 0.02, 
             f'p-value: {p_value:.2e}\nt-stat: {t_stat:.2f}', 
             transform=ax2.transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'training_rewards_mfq_{timestamp}.png', dpi=300, bbox_inches='tight')
    
    plt.show(block=True)

# Save Q-tables and create visualization
q_tables_dict = {agent: {str(state): values.tolist() 
                        for state, values in agents[agent].q_table.items()}
                 for agent in agents}

with open("mfq_q_tables.pkl", "wb") as f:
    pickle.dump(q_tables_dict, f)

# Call visualization
visualize_results(reward_history, early, late, t_stat, p_value)