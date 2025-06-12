import os
import numpy as np
from collections import defaultdict
from pettingzoo.mpe.simple_spread_v3 import parallel_env
import supersuit as ss
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime
import pickle

# Create directories for checkpoints and plots
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Enhanced configuration
N_EPISODES = 10000  # Increased to 50K
LEARNING_RATE = 0.1  # Increased to match MFQ
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_DECAY = 0.9995  # Modified for better exploration
EPSILON_MIN = 0.05  # Increased minimum epsilon
MAX_STEPS = 50  # Increased max steps
N_AGENTS = 3
ANALYSIS_WINDOW = 1000
REWARD_SCALE = 2.0  # New parameter

# Checkpoint configurations
CHECKPOINT_INTERVAL = 5000
PLOT_INTERVAL = 10000

# Helper functions
def save_checkpoint(episode, agents, reward_history, filename_prefix):
    checkpoint = {
        'episode': episode,
        'reward_history': reward_history,
        'agents': {name: {
            'q_table': dict(agent.q_table),
            'last_states': agent.last_states
        } for name, agent in agents.items()}
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"checkpoints/{filename_prefix}_checkpoint_ep{episode}_{timestamp}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"Checkpoint saved: {filename}")

def plot_progress(reward_history, episode):
    if len(reward_history) < 1000:
        return
        
    window = 1000
    smoothed = np.convolve(reward_history, np.ones(window)/window, mode='valid')
    
    plt.figure(figsize=(10, 6))
    plt.plot(smoothed)
    plt.title(f'Moving Average Reward (Window={window}) at Episode {episode}')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"plots/iql_progress_ep{episode}_{timestamp}.png"
    plt.savefig(filename)
    plt.close()

def check_learning_collapse(reward_history, window=1000):
    if len(reward_history) < window * 2:
        return False
    recent_avg = np.mean(reward_history[-window:])
    previous_avg = np.mean(reward_history[-2*window:-window])
    return recent_avg < previous_avg * 0.7

def visualize_results(reward_history, early, late, t_stat, p_value, filename='training_rewards_iql.png'):
    plt.figure(figsize=(12, 8))
    
    # Create two subplots with specific height ratios
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
    
    # Smoothed rewards using two different windows to show both trends
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
    
    # Save the plot
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {filename}")
    
    # Show the plot and close it properly
    plt.show()
    plt.close('all')

class IQLAgent:
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
        self.q_table = defaultdict(lambda: np.zeros(action_space.n))
        
        # Add experience buffer for more stable learning
        self.last_states = []
        self.buffer_size = 1000

    def get_state_key(self, obs):
        # Enhanced state discretization
        return tuple(np.round(obs, 1))  # Changed from 2 to 1 for finer granularity

    def act(self, obs):
        s = self.get_state_key(obs)
        # Optimistic initialization for unexplored states
        if s not in self.last_states:
            self.q_table[s] = np.ones(self.action_space.n) * 0.1
            if len(self.last_states) < self.buffer_size:
                self.last_states.append(s)
        
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()
        return int(np.argmax(self.q_table[s]))

    def learn(self, obs, action, reward, next_obs, done):
        s = self.get_state_key(obs)
        s_next = self.get_state_key(next_obs)
        
        # Modified reward structure to encourage exploration and cooperation
        if not done:
            reward = reward * 1.5  # Encourage longer episodes
        
        # Q-learning with eligibility traces
        best_next = np.max(self.q_table[s_next])
        td_target = reward + (0 if done else self.gamma * best_next)
        td_error = td_target - self.q_table[s][action]
        
        # Adaptive learning rate based on visited state frequency
        visit_count = self.last_states.count(s)
        adaptive_lr = self.lr / (1 + 0.1 * visit_count)
        
        self.q_table[s][action] += adaptive_lr * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

# Environment setup with reward shaping
env = parallel_env(N=N_AGENTS, local_ratio=0.7, max_cycles=MAX_STEPS)  # Increased local_ratio
env = ss.dtype_v0(env, dtype="float32")
env = ss.clip_reward_v0(env)

# Initialize agents
agents = {
    name: IQLAgent(env.observation_space(name), env.action_space(name))
    for name in env.possible_agents
}

# Training loop
reward_history = []
episode_lengths = []
best_average_reward = float('-inf')

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
            
        actions = {
            agent: agents[agent].act(obs[agent])
            for agent in active_agents if not dones[agent]
        }
        
        next_obs, rewards, terminations, truncations, _ = env.step(actions)
        
        # Enhanced reward shaping
        for agent in active_agents:
            if not dones[agent]:
                agent_pos = next_obs[agent][:2]
                cooperation_bonus = 0
                distance_penalty = 0
                
                for other_agent in active_agents:
                    if other_agent != agent:
                        other_pos = next_obs[other_agent][:2]
                        distance = np.linalg.norm(agent_pos - other_pos)
                        if 0.5 < distance < 2.0:
                            cooperation_bonus += 1.0
                        elif distance <= 0.5:
                            distance_penalty -= 1.0
                
                shaped_reward = (rewards[agent] + cooperation_bonus + distance_penalty) * REWARD_SCALE
                
                agents[agent].learn(
                    obs[agent], 
                    actions[agent], 
                    shaped_reward,
                    next_obs[agent],
                    terminations.get(agent, False)
                )
        
        obs = next_obs
        total_reward += sum(rewards.values())
    
    reward_history.append(total_reward)
    episode_lengths.append(step_count)
    
    # Checkpointing and progress monitoring
    if episode > 0 and episode % CHECKPOINT_INTERVAL == 0:
        save_checkpoint(episode, agents, reward_history, 'iql')
        
        recent_rewards = reward_history[-1000:]
        average_reward = np.mean(recent_rewards)
        
        if average_reward > best_average_reward:
            best_average_reward = average_reward
            save_checkpoint(episode, agents, reward_history, 'best_iql')
    
    if episode > 0 and episode % PLOT_INTERVAL == 0:
        plot_progress(reward_history, episode)
    
    if episode > 2000 and episode % 1000 == 0:
        if check_learning_collapse(reward_history):
            print(f"Warning: Potential learning collapse detected at episode {episode}!")
    
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

# After all the analysis, call the visualization function
early = np.array(reward_history[:ANALYSIS_WINDOW])
late = np.array(reward_history[-ANALYSIS_WINDOW:])
t_stat, p_value = stats.ttest_ind(early, late)

# Final visualization
try:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'training_rewards_iql_{timestamp}.png'
    visualize_results(reward_history, early, late, t_stat, p_value, filename=filename)
except Exception as e:
    print(f"Visualization error: {e}")
    plt.close('all')  # Ensure all plots are closed