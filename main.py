# %% Part 0 Package import
import os.path
import sys

# from gymnasium.wrappers import RecordVideo
import dsrl.offline_metadrive
import time
import datetime

import matplotlib.pyplot as plt
import torch
import itertools
import numpy as np
from loguru import logger
import argparse
from tqdm import trange

from agents.fisor_2024.config import FISOR_config, update_config
from agents.fisor_2024.agent import FISOR, FISORV1, FISORV2, FISORV3, FLOWCHUNK, FLOWCHUNKV1,\
    FLOWCHUNKWL, FLOWCHUNKWLN, FLOWCHUNKZS, FLOWCHUNKNF, FLOWNFS, FLOWNFSW, FLOWNFWF, FLOWCHUNKNFS, FLOWNFSFEASI

from utils.Buffer import data_buffer
from utils.seed import setup_seed, seed_env
from utils.Evaluation import eval_policy
from utils.torch_acceleration import (
    compile_policy_modules,
    configure_matmul_precision,
    parse_compile_dynamic,
    resolve_compile_enabled,
)

import wandb


def get_env_name(env_name0):
    """Convert short environment name to full environment name"""
    env_mapping = {
        "OfflineCarGoal1": "OfflineCarGoal1-v0",
        "OfflineCarGoal2": "OfflineCarGoal2-v0",
        "OfflinePointGoal1": "OfflinePointGoal1-v0",
        "OfflinePointGoal2": "OfflinePointGoal2-v0",
        "OfflineCarButton1": "OfflineCarButton1-v0",
        "OfflineCarButton2": "OfflineCarButton2-v0",
        "OfflinePointButton1": "OfflinePointButton1-v0",
        "OfflinePointButton2": "OfflinePointButton2-v0",
        "OfflineCarPush1": "OfflineCarPush1-v0",
        "OfflineCarPush2": "OfflineCarPush2-v0",
        "OfflinePointPush1": "OfflinePointPush1-v0",
        "OfflinePointPush2": "OfflinePointPush2-v0",
        "OfflineAntVel": "OfflineAntVelocity-v1",
        "OfflineHalfCheetahVel": "OfflineHalfCheetahVelocity-v1",
        "OfflineSwimmerVel": "OfflineSwimmerVelocity-v1",
        "OfflineHopperVel": "OfflineHopperVelocity",
        "OfflineCarRun": "OfflineCarRun-v0",
        "OfflineAntRun": "OfflineAntRun-v0",
        "OfflineDroneRun": "OfflineDroneRun-v0",
        "OfflineCarCircle": "OfflineCarCircle-v0",
        "OfflineDroneCircle": "OfflineDroneCircle-v0",
        "OfflineAntCircle": "OfflineAntCircle-v0",
        "OfflineBallCircle": "OfflineBallCircle-v0",
        "OfflineBallRun": "OfflineBallRun-v0",
        # METADRIVE
        "easysparse": "OfflineMetadrive-easysparse-v0",
        "easymean": "OfflineMetadrive-easymean-v0",
        "easydense": "OfflineMetadrive-easydense-v0",
        "mediumsparse": "OfflineMetadrive-mediumsparse-v0",
        "mediummean": "OfflineMetadrive-mediummean-v0",
        "mediumdense": "OfflineMetadrive-mediumdense-v0",
        "hardsparse": "OfflineMetadrive-hardsparse-v0",
        "hardmean": "OfflineMetadrive-hardmean-v0",
        "harddense": "OfflineMetadrive-harddense-v0"
    }
    return env_mapping.get(env_name0, env_name0)


def setup_directories(path_head, setting):
    """Create necessary directories and return file paths"""
    directories = [
        f"{path_head}results",
        f"{path_head}best_models",
        f"{path_head}checkpoint_models",
        f"artifacts/training_curve/{setting}"
    ]

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

    return {
        'eval': f"{path_head}results/{setting}",
        'best_model': f"{path_head}best_models/{setting}",
        'checkpoint': f"{path_head}checkpoint_models/{setting}",
        'curve': f"artifacts/training_curve/{setting}"
    }


def initialize_policy(algo, buffer, device, config):
    """Initialize policy based on algorithm type"""
    policy_mapping = {
        "FISOR": FISOR,
        "FISORV1": FISORV1,
        "FISORV2": FISORV2,
        "FISORV3": FISORV3,
        "FLOWCHUNK": FLOWCHUNK,
        "FLOWCHUNKV1": FLOWCHUNKV1,
        "FLOWCHUNKWL": FLOWCHUNKWL,
        "FLOWCHUNKWLN": FLOWCHUNKWLN,
        "FLOWCHUNKZS": FLOWCHUNKZS,
        "FLOWCHUNKNF": FLOWCHUNKNF,
        "FLOWNFS": FLOWNFS,
        "FLOWNFSW": FLOWNFSW,
        "FLOWNFWF": FLOWNFWF,
        "FLOWCHUNKNFS": FLOWCHUNKNFS,
        "FLOWNFSFEASI": FLOWNFSFEASI
    }

    if algo not in policy_mapping:
        raise NotImplementedError(f"No such Algorithm {algo}")

    policy_class = policy_mapping[algo]
    return policy_class(buffer.obs_dim, buffer.act_dim, buffer.max_action, device, config)


def log_evaluation_results(eval_results, gradient_step, saving_logwriter):
    """Log evaluation results to wandb"""
    if not saving_logwriter:
        return

    (avg_reward, std_reward, MAX_reward, MIN_reward,
     avg_cost, std_cost, MAX_cost, MIN_cost, _, _) = eval_results

    if saving_logwriter:
        wandb.log({
            "Evaluation/reward_mean": avg_reward,
            "Evaluation/reward_std": std_reward,
            "Evaluation/reward_max": MAX_reward,
            "Evaluation/reward_min": MIN_reward,
            "Evaluation/cost_mean": avg_cost,
            "Evaluation/cost_std": std_cost,
            "Evaluation/cost_max": MAX_cost,
            "Evaluation/cost_min": MIN_cost
        }, step=gradient_step)


def update_best_model(eval_results, best_reward, best_cost, total_train, gradient_step,
                     policy, saving_model, best_policy_path, saving_logwriter):
    """Update best model if performance improved"""
    (avg_reward, std_reward, _, _, avg_cost, _, _, _, _, _) = eval_results

    if (avg_cost < best_cost or (avg_cost == best_cost and avg_reward > best_reward)):

        best_reward = avg_reward
        best_cost = avg_cost
        corr_std = std_reward
        best_idx = gradient_step

        if saving_model:
            policy.save_model(best_policy_path)

        if saving_logwriter:
            wandb.log({
                "Best_Performance/Best_idx": best_idx,
                "Best_Performance/Best_reward": best_reward,
                "Best_Performance/Corr_std": corr_std
            }, step=gradient_step)

        return best_reward, best_cost, best_idx, True

    return best_reward, best_cost, None, False


def save_checkpoint_if_needed(total_train, checkpoint_start, checkpoint_every, checkpoint,
                             policy, saving_model, checkpoint_path, eval_results,
                             gradient_step, saving_logwriter):
    """Save checkpoint if conditions are met"""
    if total_train >= checkpoint_start and total_train % checkpoint_every == 0:
        if saving_model:
            policy.save_model(f"{checkpoint_path}_checkpoint{checkpoint}")

        if saving_logwriter:
            (avg_reward, std_reward, MAX_reward, MIN_reward, _, _, _, _, _, _) = eval_results
            wandb.log({
                "Checkpoint/reward_mean": avg_reward,
                "Checkpoint/reward_std": std_reward,
                "Checkpoint/reward_max": MAX_reward,
                "Checkpoint/reward_min": MIN_reward
            }, step=gradient_step)

        return checkpoint + 1
    return checkpoint


def plot_training_curves(eval_rewards, eval_costs, curve_path, setting):
    """Plot and save training curves"""
    if not eval_rewards or not eval_costs:
        return

    eval_reward_array = np.array(eval_rewards)
    eval_cost_array = np.array(eval_costs)

    mean_r = np.mean(eval_reward_array, axis=1)
    mean_c = np.mean(eval_cost_array, axis=1)
    max_r = np.max(eval_reward_array, axis=1)
    max_c = np.max(eval_cost_array, axis=1)
    min_r = np.min(eval_reward_array, axis=1)
    min_c = np.min(eval_cost_array, axis=1)

    steps = np.arange(len(mean_r))

    # Plot rewards
    plt.figure(figsize=(10, 6))
    plt.plot(steps, mean_r, label='Reward', color='blue')
    plt.fill_between(steps, min_r, max_r, color='blue', alpha=0.2)
    plt.xlabel('Training Episodes')
    plt.ylabel('Reward')
    plt.title(f'{setting} - Training Reward Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{curve_path}/reward_curve.png")
    plt.close()

    # Plot costs
    plt.figure(figsize=(10, 6))
    plt.plot(steps, mean_c, label='Cost', color='red')
    plt.fill_between(steps, min_c, max_c, color='red', alpha=0.2)
    plt.xlabel('Training Episodes')
    plt.ylabel('Cost')
    plt.title(f'{setting} - Training Cost Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{curve_path}/cost_curve.png")
    plt.close()


def train_baseline(args):
    """Main training function with improved structure and readability"""

    # ==================== PART 1: INITIALIZATION ====================
    print("=" * 60)
    print("STARTING TRAINING INITIALIZATION")
    print("=" * 60)

    # Parse arguments and basic setup
    algo = args.algo
    env_name0 = args.env_name
    if "Offline" in args.env_name:
        import gymnasium as gym
    else:
        import gym
    # import gymnasium as gym
    saving_model = args.save_model
    saving_logwriter = args.wandb
    seed = args.seed
    device = torch.device(args.device)
    torch_compile_enabled = resolve_compile_enabled(args.torch_compile, device)
    compile_dynamic = parse_compile_dynamic(args.compile_dynamic)
    configured_precision = configure_matmul_precision(args.matmul_precision)

    print(f"Training Configuration:")
    print(f"   Algorithm: {algo}")
    print(f"   Environment: {env_name0}")
    print(f"   Device: {device}")
    print(f"   Seed: {seed}")
    print(f"   Model Saving: {'Enabled' if saving_model else 'Disabled'}")
    print(f"   Wandb Logging: {'Enabled' if saving_logwriter else 'Disabled'}")
    print(f"   torch.compile: {torch_compile_enabled} ({args.torch_compile}, mode={args.compile_mode})")
    print(f"   Matmul Precision: {configured_precision or 'unchanged'}")

    # Configuration setup
    eval_episode = args.eval_episode
    dataset_reward_tune = 'no'
    env_name = get_env_name(env_name0)

    config = FISOR_config
    config = update_config(env_name0, config)
    if args.chunking_length is not None:
        config["chunking_length"] = int(args.chunking_length)
    if args.max_timestep is not None:
        config["max_timestep"] = int(args.max_timestep)
    if args.eval_freq is not None:
        config["eval_freq"] = int(args.eval_freq)
    if args.batch_size is not None:
        config["batch_size"] = int(args.batch_size)
    if args.cfg_guidance is not None:
        guidance_value = float(args.cfg_guidance)
        for key in ("omega", "guidance_scale", "cfg_scale", "cfg_guidance", "guidance"):
            config[key] = guidance_value
    if args.target_cost is not None:
        config["target_cost"] = float(args.target_cost)

    setting = args.run_name or f"{algo}_{env_name0}_seed{seed}"
    eval_freq = int(config['eval_freq'])
    max_timestep = int(config['max_timestep'])
    checkpoint_start = config["checkpoint_start"]
    checkpoint_every = config["checkpoint_every"]

    setup_seed(seed)
    print(f"Random seed set to {seed}")

    # ==================== PART 2: ENVIRONMENT & BUFFER SETUP ====================
    print("Setting up environment and loading dataset...")

    env = gym.make(env_name)
    dataset = env.get_dataset()
    logger.info(f"Loaded {len(dataset['rewards'])} transitions from dataset")

    buffer = data_buffer(dataset, device, dataset_reward_tune, config["normalize"])
    print("DSRL Markov datasets loaded successfully")

    eval_env = gym.make(env_name)
    eval_env.set_target_cost(args.target_cost if args.target_cost is not None else config['cost_limit'])
    seed_env(eval_env, seed)
    # eval_env = RecordVideo(
    #     eval_env,
    #     video_folder="./videos",  # 视频保存目录
    #     episode_trigger=lambda episode_id: True,  # 每个 episode 都录制
    #     name_prefix="safety_point_goal_episode"
    # )
    print("Evaluation environment configured with target cost = 10")

    # ==================== PART 3: WANDB & POLICY INITIALIZATION ====================
    if saving_logwriter:
        wandb.init(
            project="FLOWCHUNK",
            group=env_name0,
            config=config,
            tags=[env_name, f"seed{seed}"],
            name=f"{algo}_{env_name0}"
        )
        print("Wandb logging initialized")

    print(f"Initializing {algo} policy...")
    policy = initialize_policy(algo, buffer, device, config)
    compile_policy_modules(
        policy,
        enabled=torch_compile_enabled,
        mode=args.compile_mode,
        backend=args.compile_backend,
        fullgraph=args.compile_fullgraph,
        dynamic=compile_dynamic,
    )
    print("Policy initialized successfully")

    # ==================== PART 4: DIRECTORY SETUP ====================
    path_head = args.output_root.rstrip("/\\") + "/"
    paths = setup_directories(path_head, setting)
    print(f"Directories created for experiment: {setting}")

    # policy.load_model(path_head+"best_models/FLOWNFS_OfflineCarGoal1_seed7832")

    # ==================== PART 5: TRAINING LOOP ====================
    print("=" * 60)
    print("STARTING MAIN TRAINING LOOP")
    print("=" * 60)

    print(f"Training Configuration: {setting}")
    print(f"Max timesteps: {max_timestep:,}")
    print(f"Evaluation frequency: {eval_freq}")
    print(f"Target device: {device}")
    print("=" * 80)
    time.sleep(1)

    # Training variables
    eval_rewards = []
    eval_costs = []
    best_reward = -np.inf
    best_cost = np.inf
    best_idx = 0
    checkpoint = 0
    total_train = 0
    gradient_step = 0
    start_time = time.time()

    # Main training loop
    for i_episode in itertools.count(1):
        if total_train > max_timestep:
            break

        # Progress and time estimation (use print for progress info)
        progress = (total_train / max_timestep) * 100
        print(f"Episode {i_episode} | Progress: {progress:.1f}% | Steps: {total_train:,}")

        if total_train != 0:
            elapsed_time = time.time() - start_time
            remaining_time = elapsed_time * (max_timestep - total_train) / total_train / 60
            print(f"Estimated remaining time: {remaining_time:.2f} minutes")

        # Training phase (use print for training status)
        for _ in trange(eval_freq, desc=f"Training {algo}"):
            # Core training step (kept inline for performance)
            metrics = {}
            metrics.update(policy.train_critic_only(buffer, config['batch_size']))
            metrics.update(policy.train_actor_only(buffer, gradient_step, config['batch_size']))

            if gradient_step % 800 == 0 and saving_logwriter:
                wandb.log(metrics, step=gradient_step)

            gradient_step += 1

        # Evaluation phase (use logger for important evaluation results)
        logger.info("Running evaluation...")
        eval_results = eval_policy(policy, eval_env, eval_episode)

        # Log evaluation results
        log_evaluation_results(eval_results, gradient_step, saving_logwriter)

        # Extract results for readability
        (avg_reward, std_reward, max_reward, min_reward,
         avg_cost, std_cost, max_cost, min_cost, reward_buffer, cost_buffer) = eval_results

        # Use logger for important evaluation results
        logger.info(f"Evaluation Results:")
        logger.info(f"   Reward: {avg_reward:.2f} ± {std_reward:.2f} (min: {min_reward:.2f}, max: {max_reward:.2f})")
        logger.info(f"   Cost: {avg_cost:.2f} ± {std_cost:.2f} (min: {min_cost:.2f}, max: {max_cost:.2f})")

        # Update best model
        best_reward, best_cost, best_idx_new, model_updated = update_best_model(
            eval_results, best_reward, best_cost, total_train, gradient_step,
            policy, saving_model, paths['best_model'], saving_logwriter
        )

        if model_updated:
            best_idx = best_idx_new
            # Use logger for important model updates
            logger.info(f"New best model! Reward: {best_reward:.2f}, Cost: {best_cost:.2f}")

        # Save checkpoint
        checkpoint = save_checkpoint_if_needed(
            total_train, checkpoint_start, checkpoint_every, checkpoint,
            policy, saving_model, paths['checkpoint'], eval_results,
            gradient_step, saving_logwriter
        )

        # Store evaluation results
        eval_rewards.append(reward_buffer)
        eval_costs.append(cost_buffer)

        # Save evaluation data
        if saving_model:
            # Load existing data if files exist, otherwise start with empty lists
            rewards_file = paths['eval'] + '_rewards.npy'
            costs_file = paths['eval'] + '_costs.npy'

            np.save(rewards_file, eval_rewards)
            np.save(costs_file, eval_costs)

        total_train += eval_freq

    # ==================== PART 6: POST-TRAINING ====================
    print("=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    # Plot training curves
    print("Generating training curves...")
    plot_training_curves(eval_rewards, eval_costs, paths['curve'], setting)

    # Final summary (use logger for important final results)
    total_time = (time.time() - start_time) / 60
    logger.info(f"Training Summary:")
    logger.info(f"   Total training time: {total_time:.2f} minutes")
    logger.info(f"   Best reward: {best_reward:.2f}")
    logger.info(f"   Best cost: {best_cost:.2f}")
    logger.info(f"   Best step: {best_idx}")
    print(f"Results saved to: {paths['curve']}")

    # ==================== PART 7: FINAL MODEL TESTING ====================
    if args.final_test and saving_model and os.path.exists(paths['best_model']):
        print("=" * 60)
        print("TESTING BEST MODEL")
        print("=" * 60)

        # Load the best model
        print("Loading best model for final testing...")
        # Create a new policy instance for testing
        test_policy = initialize_policy(algo, buffer, device, config)
        test_policy.load_model(paths['best_model'])
        compile_policy_modules(
            test_policy,
            enabled=torch_compile_enabled,
            mode=args.compile_mode,
            backend=args.compile_backend,
            fullgraph=args.compile_fullgraph,
            dynamic=compile_dynamic,
        )
        print("Best model loaded successfully")

        # Test for 100 episodes with random seeds
        test_episodes = 60
        print(f"Running {test_episodes} test episodes with random seeds...")

        final_reward_buffer, final_cost_buffer = test_final_model(
            test_policy, env_name, config, test_episodes
        )

        # Log final test results
        logger.info("=" * 40)
        logger.info("FINAL MODEL TEST RESULTS (100 episodes with random seeds):")
        logger.info("=" * 40)
        logger.info(f"Final Test Reward: {np.mean(final_reward_buffer):.2f} STD: {np.std(final_reward_buffer):.2f}")
        logger.info(f"   Range: [{np.min(final_reward_buffer):.2f}, {np.max(final_reward_buffer):.2f}]")
        logger.info(f"Final Test Cost: {np.mean(final_cost_buffer):.2f} STD: {np.std(final_cost_buffer):.2f}")
        logger.info(f"   Range: [{np.min(final_cost_buffer):.2f}, {np.max(final_cost_buffer):.2f}]")
    else:
        print("No best model found or model saving was disabled. Skipping final test.")


def test_final_model(policy, env_name, config, test_episodes=100):
    """Test final model with random seeds for each episode"""
    print(f"Testing final model for {test_episodes} episodes with random seeds...")
    if 'metadrive' in env_name.lower():
        import gym
    else:
        import gymnasium as gym

    final_reward_buffer = []
    final_cost_buffer = []

    for i in range(test_episodes//20):
        # Create environment with random seed for each episode
        test_env = gym.make(env_name)
        test_env.set_target_cost(config.get('target_cost', config['cost_limit']))
        for ep in range(20):
            # Run one episode
            obs, _ = test_env.reset()
            done, truncated = False, False
            episode_reward = 0
            episode_cost = 0
            step_count = 0

            while not (done or truncated):
                # Use policy to select action
                action = policy.select_action_from_candidates(obs, eval=True)

                # Take action
                obs, reward, terminated, truncated, info = test_env.step(action)
                episode_reward += reward
                if 'cost_hazards' in info:
                    episode_cost += info['cost_hazards']
                elif 'cost' in info:
                    episode_cost += info['cost']
                else:
                    if 'y_velocity' not in info:
                        agent_velocity = np.abs(info['x_velocity'])
                    else:
                        agent_velocity = np.sqrt(info['x_velocity'] ** 2 + info['y_velocity'] ** 2)
                    cost = float(agent_velocity > 3.2096) # HalfCheetah
                    # cost = float(agent_velocity > 2.6222)  # Ant
                    episode_cost += cost
                step_count += 1

            _r_ep, _c_ep = test_env.get_normalized_score(episode_reward, episode_cost)
            final_reward_buffer.append(_r_ep)
            final_cost_buffer.append(_c_ep)


        test_env.close()

    return final_reward_buffer, final_cost_buffer

# OUR METHOD: FLOWNFS
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', default='FLOWNFS', type=str,
                        help="Choose from reproduced baseline algorithms ('BCQLag', 'FISOR', 'CPQ')")
    parser.add_argument('--env_name', default='OfflinePointGoal2', type=str,
                        help="Choose from mujuco domain ('halfcheetah', 'hopper', 'walker2d'), "
                             "or franka kitchen domain ('kitchen').")
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--device', default="cuda:0", type=str)
    parser.add_argument('--run-name', default=None, type=str)
    parser.add_argument('--output-root', default='artifacts/fisor_2024', type=str)
    parser.add_argument('--cfg-guidance', default=None, type=float)
    parser.add_argument('--chunking-length', default=None, type=int)
    parser.add_argument('--max-timestep', default=None, type=int)
    parser.add_argument('--eval-freq', default=None, type=int)
    parser.add_argument('--batch-size', default=None, type=int)
    parser.add_argument('--eval-episode', default=20, type=int)
    parser.add_argument('--target-cost', default=None, type=float)
    parser.add_argument('--save-model', dest='save_model', action='store_true')
    parser.add_argument('--wandb', dest='wandb', action='store_true')
    parser.add_argument('--final-test', action='store_true')
    parser.add_argument('--not_saving_model', dest='save_model', action='store_false',
                        help="Legacy flag kept for compatibility; disables model saving")
    parser.add_argument('--not_saving_logwriter', dest='wandb', action='store_false',
                        help="Legacy flag kept for compatibility; disables wandb logging")
    parser.set_defaults(save_model=False, wandb=False)
    parser.add_argument('--torch-compile', default='auto', choices=['auto', 'on', 'off'],
                        help='Compile policy networks with torch.compile. auto enables it on CUDA devices.')
    parser.add_argument('--compile-mode', default='reduce-overhead',
                        choices=['default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs'])
    parser.add_argument('--compile-backend', default='inductor', type=str)
    parser.add_argument('--compile-fullgraph', action='store_true')
    parser.add_argument('--compile-dynamic', default='auto', choices=['auto', 'true', 'false'])
    parser.add_argument('--matmul-precision', default='high', choices=['highest', 'high', 'medium', 'off'],
                        help='Float32 matmul precision. Use off to leave PyTorch defaults unchanged.')

    args = parser.parse_args()
    if args.matmul_precision == 'off':
        args.matmul_precision = None

    try:
        train_baseline(args)
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    finally:
        print("Training session ended")
