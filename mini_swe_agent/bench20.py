#!/usr/bin/env python3
"""Run 20 swebench tasks concurrently and report timing stats."""

import argparse
import asyncio
import os
import sys
import time

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from recipe.mini_swe_agent.run_standalone import (
    download_harbor_dataset,
    load_harbor_task_from_path,
    run_standalone,
)

CONCURRENCY = 20
NUM_TASKS = 20
DATASET = "terminal-bench@2.0"


async def run_one(task_path: str, instruction: str, idx: int, sem: asyncio.Semaphore, model: str):
    async with sem:
        print(f"[{idx:>2}] START")
        t0 = time.monotonic()
        try:
            _msgs, reward = await run_standalone(
                task_path=task_path,
                instruction=instruction,
                model_name=model,
                max_turns=0,
                command_timeout=30,
                verbose=False,
            )
            elapsed = time.monotonic() - t0
            print(f"[{idx:>2}] DONE  reward={reward:.1f}  time={elapsed:.1f}s")
            return {"idx": idx, "reward": reward, "time": elapsed, "error": None}
        except Exception as e:
            elapsed = time.monotonic() - t0
            print(f"[{idx:>2}] ERROR {e!r}  time={elapsed:.1f}s")
            return {"idx": idx, "reward": None, "time": elapsed, "error": str(e)}


async def main(model: str):
    import logging
    logging.basicConfig(level=logging.WARNING)
    for noisy in ["httpcore", "httpx", "urllib3", "openai", "e2b", "langsmith"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    print(f"Downloading {DATASET}...")
    downloaded = download_harbor_dataset(DATASET)
    tasks = []
    for i in range(min(NUM_TASKS, len(downloaded))):
        path = str(downloaded[i].local_path)
        _, instruction = load_harbor_task_from_path(path)
        tasks.append((path, instruction))

    print(f"Running {len(tasks)} tasks with concurrency={CONCURRENCY}, model={model}...\n")
    sem = asyncio.Semaphore(CONCURRENCY)
    t_wall_start = time.monotonic()

    results = await asyncio.gather(
        *(run_one(path, instr, i, sem, model) for i, (path, instr) in enumerate(tasks))
    )

    t_wall = time.monotonic() - t_wall_start

    # Stats
    times = [r["time"] for r in results]
    rewards = [r["reward"] for r in results if r["reward"] is not None]
    errors = [r for r in results if r["error"] is not None]

    print(f"\n{'='*60}")
    print(f"RESULTS  ({len(tasks)} tasks, concurrency={CONCURRENCY})")
    print(f"{'='*60}")
    print(f"Wall clock:  {t_wall:.1f}s")
    print(f"Time  min:   {min(times):.1f}s")
    print(f"Time  mean:  {sum(times)/len(times):.1f}s")
    print(f"Time  max:   {max(times):.1f}s")
    if rewards:
        print(f"Reward mean: {sum(rewards)/len(rewards):.2f}  ({sum(1 for r in rewards if r > 0)}/{len(rewards)} solved)")
    if errors:
        print(f"Errors:      {len(errors)}")
        for e in errors:
            print(f"  [{e['idx']}] {e['error'][:120]}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o")
    args = parser.parse_args()
    asyncio.run(main(model=args.model))
