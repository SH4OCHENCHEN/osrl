import os
import json
import csv
import time
import itertools
import argparse
from pathlib import Path

import gymnasium as gym
import dsrl
import matplotlib.pyplot as plt
import torch
import numpy as np
from loguru import logger
from tqdm import trange

from agents.fisor_2024.config import FISOR_config, update_config
from agents.fisor_2024.agent import (
    FISOR, FISORV1, FISORV2, FISORV3,
    FLOWCHUNK, FLOWCHUNKV1, FLOWCHUNKWL, FLOWCHUNKWLN, FLOWCHUNKZS,
    FLOWCHUNKNF, FLOWNFS, FLOWNFSW, FLOWNFSFEASI
)
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


def get_env_name(env_name0: str) -> str:
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
        "OfflineHalfCheetahVel": "OfflineHalfCheetahVelocity",
        "OfflineHopperVel": "OfflineHopperVelocity",
        "OfflineCarRun": "OfflineCarRun-v0",
        "OfflineAntRun": "OfflineAntRun-v0",
        "OfflineDroneRun": "OfflineDroneRun-v0",
        "OfflineCarCircle": "OfflineCarCircle-v0",
        "OfflineDroneCircle": "OfflineDroneCircle-v0",
        "OfflineAntCircle": "OfflineAntCircle-v0",
        "OfflineBallCircle": "OfflineBallCircle-v0",
        "OfflineBallRun": "OfflineBallRun-v0",
    }
    return env_mapping.get(env_name0, env_name0)


def initialize_policy(algo, buffer, device, config):
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
        "FLOWNFSFEASI": FLOWNFSFEASI,
    }
    if algo not in policy_mapping:
        raise NotImplementedError(f"No such Algorithm {algo}")
    return policy_mapping[algo](buffer.obs_dim, buffer.act_dim, buffer.max_action, device, config)


def to_jsonable(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


def infer_value(raw: str):
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none":
        return None
    try:
        if raw.startswith("0") and raw != "0" and not raw.startswith("0."):
            raise ValueError
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    try:
        return json.loads(raw)
    except Exception:
        return raw


def parse_override_items(items):
    overrides = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Expected KEY=VALUE.")
        key, value = item.split("=", 1)
        overrides[key.strip()] = infer_value(value.strip())
    return overrides


def apply_cli_overrides(config, args):
    override_notes = {}

    if args.chunking_length is not None:
        config["chunking_length"] = int(args.chunking_length)
        override_notes["chunking_length"] = int(args.chunking_length)

    if args.cfg_guidance is not None:
        guidance_value = float(args.cfg_guidance)
        # Set several common aliases so this remains robust even if the policy
        # implementation reads a slightly different config key.
        aliases = [
            "omega",
            "guidance_scale",
            "cfg_scale",
            "cfg_guidance",
            "guidance",
            "classifier_free_guidance_scale",
        ]
        for alias in aliases:
            config[alias] = guidance_value
        override_notes["cfg_guidance"] = guidance_value
        override_notes["cfg_guidance_aliases_set"] = aliases

    if args.max_timestep is not None:
        config["max_timestep"] = int(args.max_timestep)
        override_notes["max_timestep"] = int(args.max_timestep)

    if args.eval_freq is not None:
        config["eval_freq"] = int(args.eval_freq)
        override_notes["eval_freq"] = int(args.eval_freq)

    if args.batch_size is not None:
        config["batch_size"] = int(args.batch_size)
        override_notes["batch_size"] = int(args.batch_size)

    free_overrides = parse_override_items(args.override)
    for key, value in free_overrides.items():
        config[key] = value
    override_notes["manual_overrides"] = free_overrides
    return config, override_notes


def build_run_name(args):
    if args.run_name:
        return args.run_name
    parts = [args.algo, args.env_name, f"seed{args.seed}"]
    if args.cfg_guidance is not None:
        parts.append(f"cfg{args.cfg_guidance:g}")
    if args.chunking_length is not None:
        parts.append(f"h{args.chunking_length}")
    if args.tag:
        parts.append(args.tag)
    return "_".join(parts)


def setup_run_dirs(output_root: str, run_name: str):
    run_dir = Path(output_root) / run_name
    model_dir = run_dir / "models"
    checkpoint_dir = model_dir / "checkpoints"
    curve_dir = run_dir / "training_curve"
    for path in [run_dir, model_dir, checkpoint_dir, curve_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return {
        "run_dir": run_dir,
        "model_dir": model_dir,
        "checkpoint_dir": checkpoint_dir,
        "curve_dir": curve_dir,
        "best_model": model_dir / "best_model",
        "eval_csv": run_dir / "eval_metrics.csv",
        "summary_json": run_dir / "summary.json",
        "config_json": run_dir / "config_used.json",
        "reward_npy": run_dir / "eval_rewards.npy",
        "cost_npy": run_dir / "eval_costs.npy",
    }


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, ensure_ascii=False)


def append_eval_row(csv_path: Path, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def plot_training_curves(eval_rewards, eval_costs, curve_dir: Path, run_name: str):
    if not eval_rewards or not eval_costs:
        return

    reward_arr = np.array(eval_rewards)
    cost_arr = np.array(eval_costs)
    steps = np.arange(len(reward_arr))

    mean_r, min_r, max_r = reward_arr.mean(axis=1), reward_arr.min(axis=1), reward_arr.max(axis=1)
    mean_c, min_c, max_c = cost_arr.mean(axis=1), cost_arr.min(axis=1), cost_arr.max(axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(steps, mean_r, label="Reward")
    plt.fill_between(steps, min_r, max_r, alpha=0.2)
    plt.xlabel("Evaluation Index")
    plt.ylabel("Reward")
    plt.title(f"{run_name} - Training Reward Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(curve_dir / "reward_curve.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(steps, mean_c, label="Cost")
    plt.fill_between(steps, min_c, max_c, alpha=0.2)
    plt.xlabel("Evaluation Index")
    plt.ylabel("Cost")
    plt.title(f"{run_name} - Training Cost Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(curve_dir / "cost_curve.png")
    plt.close()


def update_best_model(eval_results, best_reward, best_cost, total_train, gradient_step,
                      policy, save_model, best_model_path, use_wandb):
    avg_reward, std_reward = eval_results[0], eval_results[1]
    avg_cost = eval_results[4]

    improved = False
    best_idx = None
    if total_train >= 8e4 and (avg_cost < best_cost or (avg_cost == best_cost and avg_reward > best_reward)):
        best_reward = avg_reward
        best_cost = avg_cost
        best_idx = gradient_step
        improved = True
        if save_model:
            policy.save_model(str(best_model_path))
        if use_wandb:
            wandb.log({
                "Best_Performance/Best_idx": best_idx,
                "Best_Performance/Best_reward": best_reward,
                "Best_Performance/Corr_std": std_reward,
                "Best_Performance/Best_cost": best_cost,
            }, step=gradient_step)
    return best_reward, best_cost, best_idx, improved


def save_checkpoint_if_needed(total_train, checkpoint_start, checkpoint_every, checkpoint_idx,
                              policy, save_model, checkpoint_dir, eval_results,
                              gradient_step, use_wandb):
    if total_train >= checkpoint_start and total_train % checkpoint_every == 0:
        checkpoint_path = checkpoint_dir / f"checkpoint_{checkpoint_idx:03d}"
        if save_model:
            policy.save_model(str(checkpoint_path))
        if use_wandb:
            wandb.log({
                "Checkpoint/reward_mean": eval_results[0],
                "Checkpoint/reward_std": eval_results[1],
                "Checkpoint/cost_mean": eval_results[4],
                "Checkpoint/cost_std": eval_results[5],
            }, step=gradient_step)
        return checkpoint_idx + 1
    return checkpoint_idx


def test_final_model(policy, env_name, test_episodes=100, target_cost=5):
    final_reward_buffer = []
    final_cost_buffer = []
    for _ in range(test_episodes // 20):
        test_env = gym.make(env_name)
        test_env.set_target_cost(target_cost)
        for _ep in range(20):
            obs, _ = test_env.reset()
            done, truncated = False, False
            episode_reward = 0
            episode_cost = 0
            while not (done or truncated):
                action = policy.select_action_from_candidates(obs, eval=True)
                obs, reward, terminated, truncated, info = test_env.step(action)
                done = terminated
                episode_reward += reward
                if "cost_hazards" in info:
                    episode_cost += info["cost_hazards"]
                elif "cost" in info:
                    episode_cost += info["cost"]
            norm_r, norm_c = test_env.get_normalized_score(episode_reward, episode_cost)
            final_reward_buffer.append(norm_r)
            final_cost_buffer.append(norm_c)
        test_env.close()
    return final_reward_buffer, final_cost_buffer


def train(args):
    dataset_reward_tune = "no"
    eval_episode = args.eval_episode
    env_name = get_env_name(args.env_name)
    run_name = build_run_name(args)
    save_model = args.save_model
    use_wandb = args.wandb
    device = torch.device(args.device)
    torch_compile_enabled = resolve_compile_enabled(args.torch_compile, device)
    compile_dynamic = parse_compile_dynamic(args.compile_dynamic)

    config = dict(FISOR_config)
    config = update_config(args.env_name, config)
    config, override_notes = apply_cli_overrides(config, args)
    configured_precision = configure_matmul_precision(args.matmul_precision)

    setup_seed(args.seed)

    run_paths = setup_run_dirs(args.output_root, run_name)

    logger.remove()
    logger.add(lambda msg: print(msg, end=""))
    logger.add(run_paths["run_dir"] / "python_logger.log", level="INFO")

    write_json(run_paths["config_json"], {
        "args": vars(args),
        "override_notes": override_notes,
        "resolved_env_name": env_name,
        "config": config,
    })

    print("=" * 80)
    print("START TRAINING")
    print("=" * 80)
    print(f"Run name          : {run_name}")
    print(f"Algorithm         : {args.algo}")
    print(f"Environment       : {args.env_name} -> {env_name}")
    print(f"Seed              : {args.seed}")
    print(f"Device            : {device}")
    print(f"Chunking length   : {config.get('chunking_length')}")
    cfg_echo = override_notes.get("cfg_guidance", config.get("omega", None))
    print(f"CFG guidance      : {cfg_echo}")
    print(f"Eval frequency    : {config['eval_freq']}")
    print(f"Max timesteps     : {config['max_timestep']}")
    print(f"Output directory  : {run_paths['run_dir']}")
    print(f"Save model        : {save_model}")
    print(f"Use wandb         : {use_wandb}")
    print(f"torch.compile     : {torch_compile_enabled} ({args.torch_compile})")
    print(f"Compile mode      : {args.compile_mode}")
    print(f"Matmul precision  : {configured_precision or 'unchanged'}")
    print("=" * 80)

    env = gym.make(env_name)
    dataset = env.get_dataset()
    logger.info(f"Loaded {len(dataset['rewards'])} transitions from dataset")
    buffer = data_buffer(dataset, device, dataset_reward_tune, config["normalize"])

    eval_env = gym.make(env_name)
    eval_env.set_target_cost(args.target_cost)
    seed_env(eval_env, args.seed)

    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            group=args.env_name,
            config=to_jsonable(config),
            tags=[env_name, f"seed{args.seed}", args.algo],
            name=run_name,
        )

    policy = initialize_policy(args.algo, buffer, device, config)
    compile_policy_modules(
        policy,
        enabled=torch_compile_enabled,
        mode=args.compile_mode,
        backend=args.compile_backend,
        fullgraph=args.compile_fullgraph,
        dynamic=compile_dynamic,
    )

    eval_freq = int(config["eval_freq"])
    max_timestep = int(config["max_timestep"])
    checkpoint_start = int(config["checkpoint_start"])
    checkpoint_every = int(config["checkpoint_every"])

    eval_rewards = []
    eval_costs = []
    best_reward = -np.inf
    best_cost = np.inf
    best_idx = 0
    checkpoint_idx = 0
    total_train = 0
    gradient_step = 0
    start_time = time.time()

    while total_train <= max_timestep:
        progress = 100.0 * total_train / max_timestep if max_timestep > 0 else 0.0
        print(f"\n[Train] total_train={total_train} ({progress:.2f}%) gradient_step={gradient_step}")
        if total_train > 0:
            elapsed = time.time() - start_time
            remaining_min = elapsed * (max_timestep - total_train) / total_train / 60
            print(f"Estimated remaining time: {remaining_min:.2f} min")

        for _ in trange(eval_freq, desc=f"Training {args.algo}"):
            metrics = {}
            metrics.update(policy.train_critic_only(buffer, config["batch_size"]))
            metrics.update(policy.train_actor_only(buffer, gradient_step, config["batch_size"]))
            if use_wandb and gradient_step % 800 == 0:
                wandb.log(metrics, step=gradient_step)
            gradient_step += 1

        eval_results = eval_policy(policy, eval_env, eval_episode)
        (avg_reward, std_reward, max_reward, min_reward,
         avg_cost, std_cost, max_cost, min_cost,
         reward_buffer, cost_buffer) = eval_results

        logger.info(
            f"Eval @ step {gradient_step}: reward={avg_reward:.3f}±{std_reward:.3f}, "
            f"cost={avg_cost:.3f}±{std_cost:.3f}"
        )

        if use_wandb:
            wandb.log({
                "Evaluation/reward_mean": avg_reward,
                "Evaluation/reward_std": std_reward,
                "Evaluation/reward_max": max_reward,
                "Evaluation/reward_min": min_reward,
                "Evaluation/cost_mean": avg_cost,
                "Evaluation/cost_std": std_cost,
                "Evaluation/cost_max": max_cost,
                "Evaluation/cost_min": min_cost,
            }, step=gradient_step)

        best_reward, best_cost, best_idx_new, improved = update_best_model(
            eval_results, best_reward, best_cost, total_train, gradient_step,
            policy, save_model, run_paths["best_model"], use_wandb
        )
        if improved:
            best_idx = best_idx_new
            logger.info(f"New best model: best_reward={best_reward:.3f}, best_cost={best_cost:.3f}, best_idx={best_idx}")

        checkpoint_idx = save_checkpoint_if_needed(
            total_train, checkpoint_start, checkpoint_every, checkpoint_idx,
            policy, save_model, run_paths["checkpoint_dir"], eval_results,
            gradient_step, use_wandb
        )

        eval_rewards.append(reward_buffer)
        eval_costs.append(cost_buffer)
        np.save(run_paths["reward_npy"], np.array(eval_rewards, dtype=object))
        np.save(run_paths["cost_npy"], np.array(eval_costs, dtype=object))

        row = {
            "run_name": run_name,
            "algo": args.algo,
            "env_name": args.env_name,
            "seed": args.seed,
            "cfg_guidance": override_notes.get("cfg_guidance", ""),
            "chunking_length": config.get("chunking_length", ""),
            "gradient_step": gradient_step,
            "total_train": total_train,
            "avg_reward": avg_reward,
            "std_reward": std_reward,
            "max_reward": max_reward,
            "min_reward": min_reward,
            "avg_cost": avg_cost,
            "std_cost": std_cost,
            "max_cost": max_cost,
            "min_cost": min_cost,
            "best_reward": best_reward,
            "best_cost": best_cost,
            "best_idx": best_idx,
            "elapsed_minutes": (time.time() - start_time) / 60.0,
        }
        append_eval_row(run_paths["eval_csv"], row)
        total_train += eval_freq

    plot_training_curves(eval_rewards, eval_costs, run_paths["curve_dir"], run_name)

    summary = {
        "run_name": run_name,
        "algo": args.algo,
        "env_name": args.env_name,
        "resolved_env_name": env_name,
        "seed": args.seed,
        "cfg_guidance": override_notes.get("cfg_guidance", None),
        "chunking_length": config.get("chunking_length", None),
        "best_reward": best_reward,
        "best_cost": best_cost,
        "best_idx": best_idx,
        "total_minutes": (time.time() - start_time) / 60.0,
        "output_dir": str(run_paths["run_dir"]),
    }

    if args.final_test and save_model and os.path.exists(run_paths["best_model"]):
        test_policy = initialize_policy(args.algo, buffer, device, config)
        test_policy.load_model(str(run_paths["best_model"]))
        compile_policy_modules(
            test_policy,
            enabled=torch_compile_enabled,
            mode=args.compile_mode,
            backend=args.compile_backend,
            fullgraph=args.compile_fullgraph,
            dynamic=compile_dynamic,
        )
        final_reward_buffer, final_cost_buffer = test_final_model(
            test_policy, env_name, test_episodes=args.final_test_episodes, target_cost=args.target_cost
        )
        summary["final_test_reward_mean"] = float(np.mean(final_reward_buffer))
        summary["final_test_cost_mean"] = float(np.mean(final_cost_buffer))
        summary["final_test_reward_std"] = float(np.std(final_reward_buffer))
        summary["final_test_cost_std"] = float(np.std(final_cost_buffer))

    write_json(run_paths["summary_json"], summary)
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print(json.dumps(to_jsonable(summary), indent=2, ensure_ascii=False))
    print("=" * 80)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuttal-oriented training entrypoint with CLI overrides and structured outputs.")
    parser.add_argument("--algo", default="FLOWNFS", type=str)
    parser.add_argument("--env_name", default="OfflineCarGoal1", type=str)
    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--run-name", default=None, type=str)
    parser.add_argument("--tag", default="", type=str)
    parser.add_argument("--output-root", default="artifacts/rebuttal_runs", type=str)

    parser.add_argument("--cfg-guidance", default=None, type=float)
    parser.add_argument("--chunking-length", default=None, type=int)
    parser.add_argument("--max-timestep", default=None, type=int)
    parser.add_argument("--eval-freq", default=None, type=int)
    parser.add_argument("--batch-size", default=None, type=int)
    parser.add_argument("--override", action="append", default=[], help="Additional config override, e.g. --override omega=2.0")
    parser.add_argument("--torch-compile", default="auto", choices=["auto", "on", "off"],
                        help="Compile policy networks with torch.compile. auto enables it on CUDA devices.")
    parser.add_argument("--compile-mode", default="reduce-overhead",
                        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"])
    parser.add_argument("--compile-backend", default="inductor", type=str)
    parser.add_argument("--compile-fullgraph", action="store_true")
    parser.add_argument("--compile-dynamic", default="auto", choices=["auto", "true", "false"])
    parser.add_argument("--matmul-precision", default="high", choices=["highest", "high", "medium", "off"],
                        help="Float32 matmul precision. Use off to leave PyTorch defaults unchanged.")

    parser.add_argument("--eval-episode", default=20, type=int)
    parser.add_argument("--target-cost", default=5, type=int)

    parser.add_argument("--save-model", action="store_true", help="Save best model and checkpoints")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb-project", default="FLOWHG", type=str)

    parser.add_argument("--final-test", action="store_true", help="Run final 100-episode evaluation on best model")
    parser.add_argument("--final-test-episodes", default=100, type=int)

    args = parser.parse_args()
    if args.matmul_precision == "off":
        args.matmul_precision = None

    try:
        train(args)
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as exc:
        print(f"Training failed with error: {exc}")
        raise
    finally:
        print("Training session ended")
