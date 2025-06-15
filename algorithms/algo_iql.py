import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
from scipy import stats
from pettingzoo.mpe import simple_spread_v3
parallel_env = simple_spread_v3.parallel_env
import supersuit as ss
import random
import platform
import torch
import json
from collections import defaultdict

def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_system_info():
    """Get system and environment information for reproducibility"""
    return {
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'numpy_version': np.__version__,
        'torch_version': torch.__version__,
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }

def set_global_seeds(seed):
    """Set seeds for all random number generators"""
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    return seed

def save_experiment_config(config, project_root):
    """Save experiment configuration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = os.path.join(project_root, "experiment config", f"experiment_config_iql_{timestamp}.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    return config_path

'''
def save_checkpoint(episode, agents, reward_history, filename_prefix, project_root):
    try:
        checkpoint = {
            'episode': episode,
            'reward_history': reward_history,
            'agents': {name: {
                'q_table': dict(agent.q_table),
                'last_states': agent.last_states,
                'epsilon': agent.epsilon
            } for name, agent in agents.items()}
        }
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(project_root, "checkpoints", f"{filename_prefix}_checkpoint_ep{episode}_{timestamp}.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"Checkpoint saved: {filename}")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
'''

def plot_progress(reward_history, episode, project_root):
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
    filename = os.path.join(project_root, "plots", f"iql_progress_ep{episode}_{timestamp}.png")
    plt.savefig(filename)
    plt.close()

def check_learning_collapse(reward_history, window=1000):
    if len(reward_history) < window * 2:
        return False
    recent = np.mean(reward_history[-window:])
    previous = np.mean(reward_history[-2*window:-window])
    return recent < 0.5 * previous

def visualize_results(reward_history, early, late, t_stat, p_value, filename='training_rewards_iql.png', params_text=''):
    plt.figure(figsize=(12, 8))
    gs = plt.GridSpec(2, 1, height_ratios=[1, 4])
    
    ax1 = plt.subplot(gs[0])
    ax1.text(0.05, 0.5, params_text, fontsize=9, fontfamily='monospace', verticalalignment='center')
    ax1.axis('off')
    
    ax2 = plt.subplot(gs[1])
    ax2.plot(reward_history, alpha=0.2, color='blue', label='Raw rewards')
    
    window_short = 100
    window_long = 500
    smoothed_short = np.convolve(reward_history, np.ones(window_short)/window_short, mode='valid')
    smoothed_long = np.convolve(reward_history, np.ones(window_long)/window_long, mode='valid')
    
    ax2.plot(range(window_short-1, len(smoothed_short)+window_short-1), 
             smoothed_short, color='red', linewidth=2, label=f'{window_short}-episode MA')
    ax2.plot(range(window_long-1, len(smoothed_long)+window_long-1), 
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
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {filename}")
    plt.show()
    plt.close()

class IQLAgent:
    def __init__(self, obs_space, action_space, lr, gamma, epsilon, epsilon_decay, min_epsilon):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.action_space = action_space
        self.q_table = {}
        self.last_states = []

    def get_state_key(self, obs):
        return tuple(np.round(obs, 2))

    def get_default_q_values(self):
        return np.ones(self.action_space.n) * 0.1

    def act(self, obs):
        state = self.get_state_key(obs)
        if state not in self.q_table:
            self.q_table[state] = self.get_default_q_values()
        
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()
        return int(np.argmax(self.q_table[state]))

    def learn(self, state, action, reward, next_state, done):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = self.get_default_q_values()
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = self.get_default_q_values()
        
        next_value = np.max(self.q_table[next_state_key]) if not done else 0
        td_target = reward + self.gamma * next_value
        td_error = td_target - self.q_table[state_key][action]
        self.q_table[state_key][action] += self.lr * td_error
        
        self.last_states.append(state_key)
        if len(self.last_states) > 1000:
            self.last_states.pop(0)

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

def train_iql(episodes, lr, gamma, epsilon_start,
              epsilon_decay, epsilon_min, max_steps,
              n_agents, analysis_window, reward_scale, base_reward, seed, use_shaping):
    
    # Set and record the seed
    used_seed = set_global_seeds(seed)
    print(f"Using seed: {used_seed}")
    print("IQL CONFIG:")
    print(f"Episodes: {episodes}, LR: {lr}, Gamma: {gamma}, Epsilon: {epsilon_start}")
    print(f"Eps Decay: {epsilon_decay}, Min Eps: {epsilon_min}, Steps: {max_steps}")
    print(f"Agents: {n_agents}, Reward Scale: {reward_scale}, Base Reward: {base_reward}")
    print(f"Seed: {seed}, Shaping: {use_shaping}")
    
    project_root = get_project_root()
    os.makedirs(os.path.join(project_root, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(project_root, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(project_root, 'docs'), exist_ok=True)
    
    # Create experiment configuration
    config = {
        'algorithm': 'IQL',
        'seed': used_seed,
        'hyperparameters': {
            'episodes': episodes,
            'learning_rate': lr,
            'gamma': gamma,
            'epsilon_start': epsilon_start,
            'epsilon_decay': epsilon_decay,
            'epsilon_min': epsilon_min,
            'max_steps': max_steps,
            'n_agents': n_agents,
            'analysis_window': analysis_window,
            'reward_scale': reward_scale,
            'base_reward': base_reward

        },
        'system_info': get_system_info()
    }
    
    # Save configuration
    config_path = save_experiment_config(config, project_root)
    print(f"Experiment configuration saved to: {config_path}")

    # Initialize environment
    env = parallel_env(N=n_agents, local_ratio=0.7, max_cycles=max_steps)
    env = ss.dtype_v0(env, dtype="float32")
    env = ss.clip_reward_v0(env)

    # Initialize agents
    agents = {
        name: IQLAgent(env.observation_space(name), env.action_space(name), 
                      lr, gamma, epsilon_start, epsilon_decay, epsilon_min)
        for name in env.possible_agents
    }

    reward_history = []
    episode_lengths = []
    best_average_reward = float('-inf')

    # Training loop
    for episode in range(episodes):
        if episode % 100 == 0:
            print(f"Starting episode {episode}")

        obs, _ = env.reset(seed=episode)
        dones = {agent: False for agent in env.possible_agents}
        total_reward = 0
        step_count = 0

        while not all(dones.values()) and step_count < max_steps:
            step_count += 1
            active_agents = env.agents
            if not active_agents:
                break
            
            actions = {agent: agents[agent].act(obs[agent]) 
                      for agent in active_agents if not dones[agent]}
            next_obs, rewards, terminations, truncations, _ = env.step(actions)

            # Calculate and accumulate the reward per step
            step_reward = sum(rewards.values())
            total_reward += step_reward  # Average contribution for this step

            for agent in env.possible_agents:
                dones[agent] = terminations.get(agent, False) or truncations.get(agent, False)

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
                                cooperation_bonus += 2.0
                            elif distance <= 0.5:
                                distance_penalty -= 1.0


                    if use_shaping:
                        shaped_reward = rewards[agent]
                    else:
                        shaped_reward = (rewards[agent] + cooperation_bonus + distance_penalty) * reward_scale

                    is_done = terminations.get(agent, False) or truncations.get(agent, False)
                    agents[agent].learn(obs[agent], actions[agent], shaped_reward, next_obs[agent], is_done)

            obs = next_obs

        # Store the total reward (which is now properly averaged)
        reward_history.append(total_reward)
        episode_lengths.append(step_count)

        '''if episode > 0 and episode % 10000 == 0:
            save_checkpoint(episode, agents, reward_history, 'iql', project_root)
            recent_rewards = reward_history[-1000:]
            avg_reward = np.mean(recent_rewards)
            if avg_reward > best_average_reward:
                best_average_reward = avg_reward
                save_checkpoint(episode, agents, reward_history, 'best_iql', project_root) '''

        if episode > 0 and episode % 10000 == 0:
            plot_progress(reward_history, episode, project_root)

        if episode > 10000 and episode % 5000 == 0:
            if check_learning_collapse(reward_history):
                print(f"Warning: Potential learning collapse detected at episode {episode}!")

        for agent in agents.values():
            agent.decay_epsilon()

        if episode % 100 == 0:
            print(f"Episode {episode} completed in {step_count} steps with total reward: {total_reward:.2f}")

    # Analysis
    early = np.array(reward_history[:analysis_window])
    late = np.array(reward_history[-analysis_window:])
    print("\nPerformance Analysis:")
    print(f"Early mean±std: {early.mean():.2f} ± {early.std():.2f}")
    print(f"Late mean±std: {late.mean():.2f} ± {late.std():.2f}")
    print(f"Improvement: {((late.mean() - early.mean()) / abs(early.mean()) * 100):.2f}%")
    
    t_stat, p_value = stats.ttest_ind(early, late)
    print("Statistical significance:")
    print(f"t-statistic: {t_stat:.3f}")
    if p_value < 1e-10:
        print("p-value: < 1e-10")
    else:
        print(f"p-value: {p_value:.2e}")

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(project_root, "docs", f"training_rewards_iql_{use_shaping}_{episodes}_{timestamp}.png")
        params_text = (
            f"Parameters:\n"
            f"Seed: {used_seed} | Learning Rate: {lr:<6} | Episodes: {episodes:<6}\n"
            f"Agents: {n_agents:<3} | Gamma: {gamma:<4} | Epsilon Start: {epsilon_start:<4}\n"
            f"Epsilon Decay: {epsilon_decay:<7} | Epsilon Min: {epsilon_min:<4} | Max Steps: {max_steps}"
        )
        visualize_results(reward_history, early, late, t_stat, p_value, filename=filename, params_text=params_text)
    except Exception as e:
        print(f"Visualization error: {e}")
        plt.close('all')
    
    return reward_history, used_seed, config_path