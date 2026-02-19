#!/usr/bin/env python3
"""Standalone mini SWE agent runner using mini-swe-agent v2 directly.

Uses mini-swe-agent v2's DefaultAgent + LitellmModel with a custom
HarborEnvironment that executes commands in Harbor sandboxes (E2B/Docker).
This gives us 100% identical behavior to mini-swe-agent v2: same prompts,
same tool calling, same error handling, same observation formatting.

Usage:
    E2B_API_KEY=... OPENAI_API_KEY=... python recipe/mini_swe_agent/run_standalone.py \
        --dataset swebench-verified@1.0 --task-index 0 --model gpt-4o

Requires:
    pip install mini-swe-agent harbor
"""

import argparse
import asyncio
import logging
import os
import platform
import sys
from dataclasses import dataclass, field

import yaml
from minisweagent import package_dir as _mswea_dir
from minisweagent.agents.default import DefaultAgent
from minisweagent.exceptions import Submitted
from minisweagent.models import get_model

logger = logging.getLogger(__name__)

# Load mini-swe-agent v2's tool-calling config
_mswea_config = yaml.safe_load((_mswea_dir / "config" / "mini.yaml").read_text())


# ---------------------------------------------------------------------------
# HarborEnvironment — implements mini-swe-agent v2's Environment protocol
# ---------------------------------------------------------------------------


@dataclass
class HarborEnvironmentConfig:
    env: dict = field(default_factory=lambda: {
        "PAGER": "cat",
        "MANPAGER": "cat",
        "LESS": "-R",
        "PIP_PROGRESS_BAR": "off",
        "TQDM_DISABLE": "1",
    })
    timeout: int = 120


class HarborEnvironment:
    """mini-swe-agent v2 Environment backed by a Harbor sandbox.

    Implements the Environment protocol:
        - execute(action, cwd) -> dict
        - get_template_vars() -> dict
        - serialize() -> dict
    """

    def __init__(self, sandbox, sandbox_manager, *, timeout: int = 120):
        self.sandbox = sandbox
        self.sandbox_manager = sandbox_manager
        self.config = HarborEnvironmentConfig(
            env=_mswea_config.get("environment", {}).get("env", {}),
            timeout=timeout,
        )

    def execute(self, action: dict, cwd: str = "") -> dict:
        """Execute a command in the Harbor sandbox (sync wrapper over async)."""
        command = action.get("command", "")

        # Use nest_asyncio or thread to bridge sync/async
        exec_result = _run_async(
            self.sandbox_manager.execute(
                self.sandbox, command, timeout=self.config.timeout,
            )
        )

        output = (exec_result.stdout or "") + (
            "\n" + exec_result.stderr if exec_result.stderr else ""
        )

        result = {
            "output": output,
            "returncode": exec_result.return_code,
            "exception_info": "",
        }

        # Check for submission — same logic as mini-swe-agent v2's _check_finished
        self._check_finished(result)

        return result

    def _check_finished(self, output: dict):
        """Raise Submitted if output starts with the sentinel."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if (
            lines
            and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
            and output["returncode"] == 0
        ):
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {
                        "exit_status": "Submitted",
                        "submission": submission,
                    },
                }
            )

    def get_template_vars(self, **kwargs) -> dict:
        """Return template variables for prompt rendering."""
        return {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            **self.config.env,
            **kwargs,
        }

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": {
                        "env": self.config.env,
                        "timeout": self.config.timeout,
                    }
                }
            }
        }


def _run_async(coro):
    """Run an async coroutine from sync code, even inside an existing event loop."""
    import concurrent.futures

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop — just use asyncio.run
        return asyncio.run(coro)

    # Running inside an event loop (e.g., from asyncio.run) —
    # execute in a separate thread to avoid blocking
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result()


# ---------------------------------------------------------------------------
# Task loading helpers
# ---------------------------------------------------------------------------


def download_harbor_dataset(dataset: str) -> list:
    """Download a Harbor dataset and return the list of DownloadedTask objects."""
    from harbor.dataset.client import DatasetClient
    from harbor.models.job.config import RegistryDatasetConfig
    from harbor.models.registry import RemoteRegistryInfo

    if "@" in dataset:
        name, version = dataset.split("@", 1)
    else:
        name, version = dataset, None

    config = RegistryDatasetConfig(
        registry=RemoteRegistryInfo(),
        name=name,
        version=version,
    )
    return DatasetClient().download_dataset_from_config(config)


def load_harbor_task_from_dataset(dataset: str, task_index: int) -> tuple[str, str]:
    """Load a single task from a Harbor dataset registry."""
    from harbor import Task

    downloaded_tasks = download_harbor_dataset(dataset)
    if task_index >= len(downloaded_tasks):
        raise IndexError(
            f"Task index {task_index} out of range (has {len(downloaded_tasks)} tasks)"
        )
    task_path = str(downloaded_tasks[task_index].local_path)
    return task_path, Task(task_path).instruction


def load_harbor_task_from_path(task_path: str) -> tuple[str, str]:
    """Load a single task from a local Harbor task directory."""
    from harbor import Task

    return task_path, Task(task_path).instruction


# ---------------------------------------------------------------------------
# Main standalone execution
# ---------------------------------------------------------------------------


async def run_standalone(
    task_path: str,
    instruction: str,
    model_name: str = "gpt-4o",
    max_turns: int = 30,
    command_timeout: int = 120,
    env_type: str = "e2b",
    verbose: bool = True,
) -> tuple[list, float]:
    """Run mini-swe-agent v2 on a single Harbor task.

    Returns:
        Tuple of (messages list, reward score).
    """
    from recipe.mini_swe_agent.harbor_sandbox import HarborSandboxManager

    # 1. Create Harbor sandbox
    sandbox_manager = HarborSandboxManager(env_type=env_type)

    if verbose:
        print(f"Creating {env_type} sandbox for task: {task_path}")
    sandbox = await sandbox_manager.create(task_path)

    try:
        # 2. Create mini-swe-agent v2 model (uses litellm internally)
        model_config = dict(_mswea_config.get("model", {}))
        model = get_model(model_name, config=model_config)

        # 3. Create HarborEnvironment implementing mswea's Environment protocol
        env = HarborEnvironment(
            sandbox=sandbox,
            sandbox_manager=sandbox_manager,
            timeout=command_timeout,
        )

        # 4. Create mini-swe-agent v2's DefaultAgent with config from mini.yaml
        agent_config = dict(_mswea_config.get("agent", {}))
        agent_config["step_limit"] = max_turns
        agent_config["mode"] = "yolo"  # No confirmation prompts
        agent = DefaultAgent(model=model, env=env, **agent_config)

        if verbose:
            print(f"Starting agent loop (max {max_turns} turns, model={model_name})...")

        # 5. Run the agent — this is mini-swe-agent v2's actual loop
        result = agent.run(task=instruction)

        if verbose:
            exit_status = result.get("exit_status", "Unknown")
            print(f"\nAgent finished: {exit_status}")
            _print_trajectory(agent.messages)

        # 6. Compute reward
        if verbose:
            print("\nRunning verification tests...")
        reward = await sandbox_manager.compute_reward(sandbox)
        if verbose:
            print(f"Reward: {reward}")

        return agent.messages, reward

    finally:
        if verbose:
            print("Destroying sandbox...")
        await sandbox_manager.destroy(sandbox)


def _print_trajectory(messages: list[dict]):
    """Print summary of the agent trajectory."""
    turn = 0
    for msg in messages:
        role = msg.get("role", "")
        if role == "system":
            print(f"\n{'='*60}")
            print(f"SYSTEM PROMPT ({len(msg.get('content', ''))} chars)")
            print(f"{'='*60}")
        elif role == "user":
            print(f"\n{'='*60}")
            print(f"USER ({len(msg.get('content', ''))} chars)")
            print(f"{'='*60}")
            content = msg.get("content", "")
            if len(content) > 500:
                print(content[:500] + "...")
            else:
                print(content)
        elif role == "assistant":
            turn += 1
            print(f"\n{'='*60}")
            print(f"TURN {turn}")
            print(f"{'='*60}")
            if msg.get("content"):
                print(msg["content"])
            for action in msg.get("extra", {}).get("actions", []):
                print(f"\n> COMMAND: {action.get('command', '')}")
        elif role == "tool":
            print(f"\n--- Observation ---")
            content = msg.get("content", "")
            if len(content) > 1000:
                print(content[:1000] + "... (truncated)")
            else:
                print(content)
        elif role == "exit":
            print(f"\n{'='*60}")
            print(f"EXIT: {msg.get('extra', {}).get('exit_status', 'Unknown')}")
            print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Run mini-swe-agent v2 on a Harbor task.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python recipe/mini_swe_agent/run_standalone.py \\
      --dataset swebench-verified@1.0 --task-index 0 --model gpt-4o

  python recipe/mini_swe_agent/run_standalone.py \\
      --task-path /path/to/task --model gpt-4o-mini
""",
    )

    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--dataset", type=str, help="Harbor dataset identifier")
    task_group.add_argument(
        "--task-path", type=str, help="Path to a local Harbor task directory"
    )

    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--command-timeout", type=int, default=120)
    parser.add_argument(
        "--env-type",
        type=str,
        default="e2b",
        choices=["e2b", "docker", "daytona"],
    )
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    log_level = logging.INFO if not args.quiet else logging.WARNING
    logging.basicConfig(
        level=log_level, format="%(asctime)s %(name)s %(levelname)s: %(message)s"
    )
    for noisy in ["httpcore", "httpx", "urllib3", "openai", "e2b", "langsmith"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.dataset:
        task_path, instruction = load_harbor_task_from_dataset(
            args.dataset, args.task_index
        )
    else:
        task_path, instruction = load_harbor_task_from_path(args.task_path)

    messages, reward = asyncio.run(
        run_standalone(
            task_path=task_path,
            instruction=instruction,
            model_name=args.model,
            max_turns=args.max_turns,
            command_timeout=args.command_timeout,
            env_type=args.env_type,
            verbose=not args.quiet,
        )
    )

    sys.exit(0 if reward > 0 else 1)


if __name__ == "__main__":
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    main()
