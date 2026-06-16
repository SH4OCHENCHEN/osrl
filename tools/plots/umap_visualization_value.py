# %% Part 0 Package import
import os.path
import sys

import gymnasium as gym
# from gymnasium.wrappers import RecordVideo
import dsrl
import time
import datetime

import matplotlib.pyplot as plt
import torch
import itertools
import numpy as np
from loguru import logger
import argparse
from tqdm import trange
import umap
from sklearn.preprocessing import StandardScaler

from agents.fisor_2024.config import FISOR_config, update_config
from agents.fisor_2024.agent import FISOR, FISORV1, FISORV2, FISORV3, FLOWCHUNK, FLOWCHUNKV1,\
    FLOWCHUNKWL, FLOWCHUNKWLN, FLOWCHUNKZS, FLOWCHUNKNF, FLOWNFS, FLOWCHUNKNFS

from utils.Buffer import data_buffer
from utils.seed import setup_seed, seed_env
from utils.Evaluation import eval_policy

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
        "OfflineBallRun": "OfflineBallRun-v0"
    }
    return env_mapping.get(env_name0, env_name0)

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
        "FLOWCHUNKNFS": FLOWCHUNKNFS
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


def umap_visualize(args):
    env_name = 'OfflineDroneRun'
    saving_model = args.not_saving_model
    device = torch.device('cuda')

    # Configuration setup
    eval_episode = 20
    dataset_reward_tune = 'no'
    env_name = get_env_name(env_name)

    config = FISOR_config
    config = update_config(env_name, config)

    env = gym.make(env_name)
    dataset = env.get_dataset()
    logger.info(f"Loaded {len(dataset['rewards'])} transitions from dataset")

    buffer = data_buffer(dataset, device, dataset_reward_tune, config["normalize"])
    print("DSRL Markov datasets loaded successfully")

    policy1 = initialize_policy('FLOWCHUNKWLN', buffer, device, config)
    policy2 = initialize_policy('FLOWNFS', buffer, device, config)
    # policy2.load_model('artifacts/fisor_2024/checkpoint_models/FLOWNFS_OfflineBallRun_seed3036_checkpoint0')
    # policy1.load_model('artifacts/fisor_2024/checkpoint_models/FLOWCHUNKWLN_OfflineBallRun_123_checkpoint0')
    policy2.load_model('artifacts/fisor_2024/checkpoint_models/FLOWNFS_OfflineDroneRun_seed7282_checkpoint0')
    policy1.load_model('artifacts/fisor_2024/checkpoint_models/FLOWCHUNKWLN_OfflineDroneRun_123_checkpoint0')

    indices = np.random.choice(buffer.obs.shape[0], size=5000, replace=False)
    obs = buffer.obs[indices].to(device)
    vr = policy1.get_vr(obs).detach().cpu().numpy()
    vc = policy1.get_vc(obs).detach().cpu().numpy()
    cvr = policy2.get_vr(obs).detach().cpu().numpy()
    obs_ = obs.detach().cpu().numpy()

    return create_umap_visualization(obs_, vr, vc, cvr, save_path=f'umap_visualization_{env_name}.pdf')


def create_umap_visualization(obs_, vr, vc, cvr, save_path=None, figsize=(24, 8),
                             n_neighbors=15, min_dist=0.8, random_state=42,
                             fontsize=22):
    """
    Create UMAP visualization with three subplots showing vr, vc, cvr as heatmaps

    Args:
        obs_: observations data (n_samples, n_features)
        vr: value function rewards (n_samples,)
        vc: value function costs (n_samples,)
        cvr: corrected value function rewards (n_samples,)
        save_path: path to save the figure
        figsize: figure size
        n_neighbors: UMAP parameter
        min_dist: UMAP parameter
        random_state: random seed for reproducibility
        fontsize: unified font size for all text elements
    """
    print("Starting UMAP dimensionality reduction...")

    # Standardize the observations
    scaler = StandardScaler()
    obs_scaled = scaler.fit_transform(obs_)

    # Apply UMAP
    umap_reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=random_state,
        verbose=True
    )

    embedding = umap_reducer.fit_transform(obs_scaled)
    print(f"UMAP completed. Embedding shape: {embedding.shape}")

    # Create the figure with larger, more compact layout
    fig = plt.figure(figsize=figsize)

    # Create a tighter grid layout with more space for plots and less for colorbars
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1],
                         left=0.03, right=0.95, top=0.95, bottom=0.05,
                         wspace=0.25, hspace=0.15)

    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    # Define the value arrays and their corresponding titles
    values = [vr.flatten(), vc.flatten(), cvr.flatten()]
    titles = ['$V_r$', '$V_c$', r'$\bar{V}_r$']
    cmaps = ['viridis', 'plasma', 'coolwarm']

    for i, (ax, val, title, cmap) in enumerate(zip(axes, values, titles, cmaps)):
        # Create scatter plot with larger points for better visibility
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1],
                           c=val, cmap=cmap, s=20, alpha=0.85, edgecolors='none')

        # Add colorbar with smaller height (reduced shrink value)
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=15, pad=0.01)
        # cbar.set_label(['$V_r$', '$V_c$', r'$\bar{V}_r$'][i],
        #               fontsize=fontsize, labelpad=12)
        cbar.ax.tick_params(labelsize=fontsize, width=1.0, length=4)

        # Make colorbar outline black and thicker
        cbar.outline.set_linewidth(1.0)
        cbar.outline.set_edgecolor('black')

        # Set labels and title with unified font
        ax.set_xlabel('UMAP Dimension 1', fontsize=fontsize, labelpad=8)
        ax.set_ylabel('UMAP Dimension 2', fontsize=fontsize, labelpad=8)
        ax.set_title(title, fontsize=fontsize, pad=12)

        # Improve grid appearance
        ax.grid(True, alpha=0.4, linewidth=0.6)
        ax.set_axisbelow(True)

        # Set tick parameters with outward-pointing ticks
        ax.tick_params(axis='both', which='major', labelsize=fontsize,
                      width=1.0, length=6, pad=6, direction='out')

        # Set all spines to black and make them visible
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_edgecolor('black')

        # Remove statistics text box (commented out)
        # stats_text = f'Mean: {val.mean():.3f}\nStd: {val.std():.3f}\nMin: {val.min():.3f}\nMax: {val.max():.3f}'
        # ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
        #        verticalalignment='top',
        #        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.95,
        #                 edgecolor='black', linewidth=1.0),
        #        fontsize=fontsize)

        # Set aspect ratio to be equal for better visualization
        ax.set_aspect('equal', adjustable='box')


    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"UMAP visualization saved to: {save_path}")
        plt.close()
    else:
        plt.show()

    return embedding


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--not_saving_model", action="store_false", default=True)
    parser.add_argument("--env_name", type=str, default="DroneRun")
    parser.add_argument("--policy1", type=str, default="FLOWCHUNKWL")
    parser.add_argument("--policy2", type=str, default="FLOWNFS")
    parser.add_argument("--model1_path", type=str,
                        default="artifacts/fisor_2024/checkpoint_models/FLOWCHUNKWL_OfflineDroneRun_123_checkpoint0")
    parser.add_argument("--model2_path", type=str,
                        default="artifacts/fisor_2024/checkpoint_models/FLOWNFS_OfflineDroneRun_seed7558_checkpoint0")
    parser.add_argument("--sample_size", type=int, default=5000)
    parser.add_argument("--save_dir", type=str, default="dataset_distribution")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Create save directory
    # os.makedirs(args.save_dir, exist_ok=True)

    # Set random seed
    setup_seed(args.seed)

    print("Starting UMAP visualization...")
    print(f"Environment: {args.env_name}")
    print(f"Policy 1: {args.policy1}")
    print(f"Policy 2: {args.policy2}")
    print(f"Sample size: {args.sample_size}")

    embedding = umap_visualize(args)
    print("UMAP visualization completed successfully!")
