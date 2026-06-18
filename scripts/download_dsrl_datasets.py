import argparse
import sys
from pathlib import Path

import dsrl  # noqa: F401
import gymnasium as gym

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.datasets import ensure_dsrl_dataset


DEFAULT_TASKS = [
    "OfflineCarGoal1-v0",
    "OfflineCarGoal2-v0",
    "OfflinePointGoal1-v0",
    "OfflinePointGoal2-v0",
    "OfflineCarButton1-v0",
    "OfflineCarButton2-v0",
    "OfflinePointButton1-v0",
    "OfflinePointButton2-v0",
    "OfflineCarPush1-v0",
    "OfflineCarPush2-v0",
    "OfflinePointPush1-v0",
    "OfflinePointPush2-v0",
    "OfflineCarRun-v0",
    "OfflineAntRun-v0",
    "OfflineDroneRun-v0",
    "OfflineCarCircle-v0",
    "OfflineDroneCircle-v0",
    "OfflineAntCircle-v0",
    "OfflineBallCircle-v0",
    "OfflineBallRun-v0",
]


def main():
    parser = argparse.ArgumentParser(description="Download DSRL datasets from Hugging Face.")
    parser.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    parser.add_argument("--repo-id", default="YYY-45/DSRL")
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--dataset-dir", default=str(Path.home() / ".dsrl" / "datasets"))
    args = parser.parse_args()

    for task in args.tasks:
        print(f"\n==> {task}")
        env = gym.make(task)
        local_path = ensure_dsrl_dataset(
            env,
            download="hf",
            repo_id=args.repo_id,
            endpoint=args.endpoint,
            local_dir=args.dataset_dir,
        )
        env.close()
        print(f"Saved: {local_path}")


if __name__ == "__main__":
    main()
