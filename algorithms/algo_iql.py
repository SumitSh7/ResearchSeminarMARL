import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats
import csv
from pettingzoo.mpe import simple_spread_v3
parallel_env = simple_spread_v3.parallel_env
import supersuit as ss
import random
import platform
import torch

def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def set_global_seeds(seed):
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    return seed

class IQLAgent:
    def __init__(self, obs_space, action_space, lr, gamma, epsilon, epsilon_decay, min_epsilon):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.action_space = action_space
        self.q_table = {}
        
    def get_state_key(self, obs):
        return tuple(np.round(obs, 3))
        
    def get_default_q_values(self):
        return np.zeros(self.action_space.n)

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

        if not done:
            next_max = np.max(self.q_table[next_state_key])
        else:
            next_max = 0
            
        td_target = reward + self.gamma * next_max
        td_error = td_target - self.q_table[state_key][action]
        self.q_table[state_key][action] += self.lr * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

def visualize_results(reward_history, early, late, t_stat, p_value, filename='', params_text=''):
    plt.figure(figsize=(12, 8))
    gs = plt.GridSpec(2, 1, height_ratios=[1, 4])
    
    ax1 = plt.subplot(gs[0])
    ax1.text(0.05, 0.5, params_text, fontsize=9, fontfamily='monospace', verticalalignment='center')
    ax1.axis('off')
    
    ax2 = plt.subplot(gs[1])
    '''ax2.plot(reward_history, alpha=0.2, color='blue', label='Raw rewards')'''
    
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
    plt.savefig(filename, bbox_inches='tight', format='pdf')
    print(f"Plot saved as {filename}")
    plt.show()
    plt.close()

def train_iql(episodes, lr, gamma, epsilon_start,
              epsilon_decay, epsilon_min, max_steps,
              n_agents, analysis_window, reward_scale,
              base_reward=1.0, seed=None, use_shaping=True):
    
    used_seed = set_global_seeds(seed)
    print(f"Using seed: {used_seed}")
    
    project_root = get_project_root()
    results_dir = os.path.join(project_root, "results", "iql")
    os.makedirs(results_dir, exist_ok=True)
    
    env = parallel_env(N=n_agents, local_ratio=0.7, max_cycles=max_steps)
    env = ss.dtype_v0(env, dtype="float32")
    env = ss.clip_reward_v0(env)
    
    agents = {
        name: IQLAgent(env.observation_space(name), env.action_space(name),
                      lr, gamma, epsilon_start, epsilon_decay, epsilon_min)
        for name in env.possible_agents
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(results_dir, f"IQL_{episodes}_{lr}_{epsilon_decay}_{timestamp}.csv")
    
    reward_history = []
    result_buffer = []
    buffer_size = 100
    
    try:
        for episode in range(episodes):
            if episode % 100 == 0:
                print(f"Starting episode {episode}")
                
            obs, _ = env.reset(seed=episode)
            dones = {agent: False for agent in env.possible_agents}
            total_reward = 0
            step = 0
            
            while not all(dones.values()) and step < max_steps:
                step += 1
                actions = {}
                
                for agent in env.agents:
                    if not dones[agent]:
                        actions[agent] = agents[agent].act(obs[agent])
                
                next_obs, rewards, terminations, truncations, _ = env.step(actions)
                
                episode_reward = 0
                for agent in env.agents:
                    if not dones[agent]:
                        shaped_reward = rewards[agent]
                        if use_shaping:
                            agent_pos = next_obs[agent][:2]
                            min_distance = float('inf')
                            for other in env.agents:
                                if other != agent:
                                    other_pos = next_obs[other][:2]
                                    distance = np.linalg.norm(agent_pos - other_pos)
                                    min_distance = min(min_distance, distance)
                            
                            if min_distance < 0.5:
                                shaped_reward -= 1.0
                            elif min_distance < 2.0:
                                shaped_reward += 2.0 * (2.0 - min_distance)
                        
                        shaped_reward *= reward_scale
                        agents[agent].learn(obs[agent], actions[agent], 
                                         shaped_reward, next_obs[agent], 
                                         terminations[agent])
                        
                        episode_reward += rewards[agent]
                
                total_reward += episode_reward
                obs = next_obs
                
                for agent in env.agents:
                    dones[agent] = terminations.get(agent, False) or truncations.get(agent, False)
            
            for agent in agents.values():
                agent.decay_epsilon()
            
            reward_history.append(total_reward)
            result_buffer.append([episode, total_reward])
            
            if len(result_buffer) >= buffer_size:
                with open(csv_filename, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(result_buffer)
                result_buffer = []
                
            if episode % 100 == 0:
                recent_rewards = reward_history[-100:]
                avg_reward = sum(recent_rewards) / len(recent_rewards)
                print(f"Episode {episode}: Avg Reward (last 100): {avg_reward:.2f}, "
                      f"Epsilon: {list(agents.values())[0].epsilon:.3f}")
                
    finally:
        env.close()
        if result_buffer:
            with open(csv_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(result_buffer)
    
    early = np.array(reward_history[:analysis_window])
    late = np.array(reward_history[-analysis_window:])
    print("\nPerformance Analysis:")
    print(f"Early mean±std: {early.mean():.2f} ± {early.std():.2f}")
    print(f"Late mean±std: {late.mean():.2f} ± {late.std():.2f}")
    print(f"Improvement: {((late.mean() - early.mean()) / abs(early.mean()) * 100):.2f}%")

    t_stat, p_value = stats.ttest_ind(early, late)
    print("Statistical significance:")
    print(f"t-statistic: {t_stat:.3f}")
    print(f"p-value: {'< 1e-10' if p_value < 1e-10 else f'{p_value:.2e}'}")

    try:
        # Set specific path and filename
        plots_dir = os.path.join(project_root, "plots")
        os.makedirs(plots_dir, exist_ok=True)  # Ensure directory exists
        filename = os.path.join(plots_dir, f"IQL_{episodes}_{lr}_{epsilon_decay}_{timestamp}.pdf")
        
        params_text = (
            f"Parameters:\n"
            f"Seed: {used_seed} | Learning Rate: {lr:<6} | Episodes: {episodes:<6}\n"
            f"Agents: {n_agents:<3} | Gamma: {gamma:<4} | Epsilon Start: {epsilon_start:<4}\n"
            f"Epsilon Decay: {epsilon_decay:<7} | Epsilon Min: {epsilon_min:<4} | Max Steps: {max_steps}"
        )
        visualize_results(reward_history, early, late, t_stat, p_value, 
                         filename=filename, params_text=params_text)
    except Exception as e:
        print(f"Visualization error: {e}")
        plt.close('all')
    
    return reward_history, used_seed, csv_filename