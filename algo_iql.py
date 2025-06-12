import numpy as np
from collections import defaultdict
from pettingzoo.mpe.simple_spread_v3 import parallel_env
import supersuit as ss
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime

# Enhanced configuration
N_EPISODES = 20000  # Doubled from 10000 to 20000
LEARNING_RATE = 0.05  # Increased from 0.01 to 0.05
GAMMA = 0.99  # Keep this as is since it's already good for long-term planning
EPSILON_START = 1.0
EPSILON_DECAY = 0.9998  # Slightly slower decay since we have more episodes
EPSILON_MIN = 0.01
MAX_STEPS = 25
N_AGENTS = 3
ANALYSIS_WINDOW = 1000  # Increased due to more episodes

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
        
        # Reward shaping: encourage agents to stay active and cooperate
        for agent in active_agents:
            if not dones[agent]:
                # Add cooperation bonus based on distance to other agents
                agent_pos = next_obs[agent][:2]
                cooperation_bonus = 0
                for other_agent in active_agents:
                    if other_agent != agent:
                        other_pos = next_obs[other_agent][:2]
                        distance = np.linalg.norm(agent_pos - other_pos)
                        if 0.5 < distance < 2.0:  # Sweet spot for cooperation
                            cooperation_bonus += 0.5
                
                shaped_reward = rewards[agent] + cooperation_bonus
                
                # Learning step with shaped reward and done flag
                agents[agent].learn(
                    obs[agent], 
                    actions[agent], 
                    shaped_reward,
                    next_obs[agent],
                    terminations.get(agent, False)
                )
        
        # Update dones combining terminations and truncations
        for agent in env.possible_agents:
            dones[agent] = terminations.get(agent, False) or truncations.get(agent, False)
        
        obs = next_obs
        total_reward += sum(rewards.values())
    
    if episode % 100 == 0:
        print(f"Episode {episode} completed in {step_count} steps with total reward: {total_reward}")
    
    # Decay epsilon for all agents
    for agent in agents.values():
        agent.decay_epsilon()
    
    reward_history.append(total_reward)
    episode_lengths.append(step_count)

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
    
    # Add event handler for window closing
    def handle_close(evt):
        plt.close('all')
    
    fig.canvas.mpl_connect('close_event', handle_close)
    
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
    
    # Plot raw rewards with low alpha for transparency
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
    
    # Add horizontal lines for early and late performance
    ax2.axhline(y=early.mean(), color='orange', linestyle='--', alpha=0.5, 
               label=f'Early mean: {early.mean():.2f}')
    ax2.axhline(y=late.mean(), color='purple', linestyle='--', alpha=0.5,
               label=f'Late mean: {late.mean():.2f}')
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Reward')
    ax2.set_title(f'Training Progress over {N_EPISODES} Episodes\nImprovement: {((late.mean() - early.mean()) / abs(early.mean()) * 100):.2f}%')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add statistical information as text
    ax2.text(0.02, 0.02, 
             f'p-value: {p_value:.2e}\nt-stat: {t_stat:.2f}', 
             transform=ax2.transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the plot with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'training_rewards_iql_{timestamp}.png', dpi=300, bbox_inches='tight')
    
    plt.show(block=True)  # This will block until the window is closed

# After all the analysis, call the visualization function
visualize_results(reward_history, early, late, t_stat, p_value)