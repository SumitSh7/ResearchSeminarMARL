import sys
import os
import argparse
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from algorithms import train_iql

def main():
    parser = argparse.ArgumentParser(description="Run IQL training")

    # Core hyperparameters
    parser.add_argument("--episodes", type=int, default=2000, help="Number of training episodes")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--seed", type=int, default=424242427, help="Random seed for reproducibility")

    # Epsilon schedule
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="Starting value for epsilon")
    parser.add_argument("--epsilon_decay", type=float, default=0.9997, help="Epsilon decay rate")
    parser.add_argument("--epsilon_min", type=float, default=0.05, help="Minimum value for epsilon")

    # Environment and learning settings
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum number of steps per episode")
    parser.add_argument("--n_agents", type=int, default=3, help="Number of agents in the environment")
    parser.add_argument("--analysis_window", type=int, default=200, help="Window size for analysis")
    parser.add_argument("--base_reward", type=float, default=0.2, help="Base reward value")
    parser.add_argument("--use_shaping", type=bool, default=False, help="Shaped reward value")
    parser.add_argument("--reward_scale", type=float, default=2.0, help="Reward scale factor")
    
    # Configuration loading
    parser.add_argument("--load_config", type=str, help="Path to configuration file to load parameters from")

    args = parser.parse_args()

    if args.load_config:
        with open(args.load_config, 'r') as f:
            config = json.load(f)
            hyperparams = config['hyperparameters']
            seed = config['seed']
            # Update args with loaded config
            for key, value in hyperparams.items():
                setattr(args, key, value)
            args.seed = seed
            print(f"Loaded configuration from {args.load_config}")

    reward_history, used_seed, config_path = train_iql(
        episodes=args.episodes,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        max_steps=args.max_steps,
        n_agents=args.n_agents,
        analysis_window=args.analysis_window,
        reward_scale=args.reward_scale,
        base_reward=args.base_reward,
        use_shaping=args.use_shaping,
        seed=args.seed
    )

    print(f"\nExperiment completed:")
    print(f"Seed used: {used_seed}")
    print(f"Configuration saved to: {config_path}")

if __name__ == '__main__':
    main()