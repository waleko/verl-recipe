#!/usr/bin/env python3
"""Harbor tasks -> verl parquet converter.

Loads Harbor dataset tasks (swebench-verified, swesmith, etc.), extracts
task instructions and metadata, and converts to verl's expected parquet
format for RL training.

Usage:
    # Download dataset first:
    harbor datasets download swebench-verified@1.0

    # Convert to verl format:
    python recipe/mini_swe_agent/create_dataset.py \
        --dataset swebench-verified@1.0 \
        --output data/swebench

    # Limit number of tasks (for testing):
    python recipe/mini_swe_agent/create_dataset.py \
        --dataset swesmith@1.0 \
        --max 10 \
        --output data/swesmith_small

Requires:
    pip install harbor pandas
"""

import argparse
import os
import sys

# Ensure recipe package is importable when running as a script
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import pandas as pd

from recipe.mini_swe_agent.run_standalone import download_harbor_dataset

# Load the system prompt from mini-swe-agent's tool-calling config
from minisweagent import package_dir as _mswea_dir
import yaml as _yaml
_default_config = _yaml.safe_load((_mswea_dir / "config" / "mini.yaml").read_text())
SYSTEM_PROMPT = _default_config["agent"]["system_template"]


def generate_data(
    dataset: str,
    max_tasks: int | None = None,
    split: str = "train",
    agent_name: str = "mini_swe_agent",
) -> pd.DataFrame:
    """Load Harbor tasks and convert to verl dataset format.

    Args:
        dataset: Harbor dataset identifier (e.g. "swebench-verified@1.0").
        max_tasks: Maximum number of tasks to include (None = all).
        split: Dataset split label ("train" or "test").
        agent_name: Agent name for verl routing.

    Returns:
        DataFrame in verl's expected format.
    """
    from harbor import Task

    downloaded_tasks = download_harbor_dataset(dataset)

    if max_tasks is not None:
        downloaded_tasks = downloaded_tasks[:max_tasks]

    # Extract the dataset name without version for data_source
    data_source = dataset.split("@")[0] if "@" in dataset else dataset

    rl_dataset = {
        "prompt": [],
        "data_source": [],
        "ability": [],
        "reward_model": [],
        "extra_info": [],
        "agent_name": [],
    }

    for idx, dt in enumerate(downloaded_tasks):
        task_path = str(dt.local_path)
        try:
            task = Task(task_path)
        except Exception as e:
            print(f"Warning: Skipping task at {task_path}: {e}")
            continue

        prompt_with_template = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task.instruction},
        ]

        rl_dataset["prompt"].append(prompt_with_template)
        rl_dataset["data_source"].append(data_source)
        rl_dataset["ability"].append("swe")
        rl_dataset["reward_model"].append({"style": "rule", "ground_truth": {}})
        rl_dataset["extra_info"].append({
            "index": idx,
            "task_id": task.name,
            "task_path": task_path,
            "split": split,
        })
        rl_dataset["agent_name"].append(agent_name)

    df = pd.DataFrame(data=rl_dataset)
    print(f"Created {len(df)} samples from {dataset} ({split})")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Harbor tasks to verl parquet format",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="swebench-verified@1.0",
        help="Harbor dataset identifier (default: swebench-verified@1.0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/swebench",
        help="Output directory for parquet files (default: data/swebench)",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Maximum number of tasks to include (default: all)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Fraction of tasks for training (default: 0.9)",
    )
    parser.add_argument(
        "--agent-name",
        type=str,
        default="mini_swe_agent",
        help="Agent name for verl routing (default: mini_swe_agent)",
    )

    args = parser.parse_args()

    full_dataset = generate_data(
        dataset=args.dataset,
        max_tasks=args.max,
        agent_name=args.agent_name,
    )

    if len(full_dataset) == 0:
        print("Error: No tasks were loaded. Check your dataset.")
        exit(1)

    # Split into train/test
    split_idx = int(len(full_dataset) * args.train_ratio)
    train_dataset = full_dataset.iloc[:split_idx].copy()
    test_dataset = full_dataset.iloc[split_idx:].copy()

    # Update split labels
    train_dataset["extra_info"] = train_dataset["extra_info"].apply(
        lambda x: {**x, "split": "train"}
    )
    test_dataset["extra_info"] = test_dataset["extra_info"].apply(
        lambda x: {**x, "split": "test"}
    )

    # Save
    os.makedirs(args.output, exist_ok=True)
    train_path = os.path.join(args.output, "train.parquet")
    test_path = os.path.join(args.output, "test.parquet")

    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)

    print(f"Saved {len(train_dataset)} train samples to {train_path}")
    print(f"Saved {len(test_dataset)} test samples to {test_path}")
