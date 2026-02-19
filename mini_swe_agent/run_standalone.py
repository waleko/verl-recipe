#!/usr/bin/env python3
"""Standalone mini SWE agent — Phase A deliverable.

Runs a LangGraph agent loop aligned with mini-swe-agent v2:
- Same system/instance prompts (loaded from mini-swe-agent's mini.yaml)
- Same bash tool definition (tool-calling, not text-based code blocks)
- Same observation template (JSON format, truncation warning)
- Same submission detection (COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT)

The LangGraph StateGraph built here is reused in swe_agent_loop.py for verl
RL training — just swap ChatOpenAI for verl's ChatModel.

Usage:
    E2B_API_KEY=... OPENAI_API_KEY=... python recipe/mini_swe_agent/run_standalone.py \
        --dataset swesmith@1.0 --task-index 0 --model gpt-4o

Requires:
    pip install mini-swe-agent langchain-openai langgraph harbor
"""

import argparse
import asyncio
import logging
import os
import sys
from typing import Any, Literal

import yaml
from jinja2 import StrictUndefined, Template
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import END, MessagesState, StateGraph

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load prompts from mini-swe-agent v2's tool-calling config
# ---------------------------------------------------------------------------

from minisweagent import package_dir as _mswea_dir

_default_config = yaml.safe_load((_mswea_dir / "config" / "mini.yaml").read_text())

SYSTEM_TEMPLATE = _default_config["agent"]["system_template"]
INSTANCE_TEMPLATE = _default_config["agent"]["instance_template"]
OBSERVATION_TEMPLATE = _default_config["model"]["observation_template"]
FORMAT_ERROR_TEMPLATE = _default_config["model"]["format_error_template"]

SUBMIT_SENTINEL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
MAX_FORMAT_RETRIES = 3

# Environment variables set inside the sandbox (matches mini-swe-agent v2)
SANDBOX_ENV_VARS = _default_config.get("environment", {}).get("env", {
    "PAGER": "cat",
    "MANPAGER": "cat",
    "LESS": "-R",
    "PIP_PROGRESS_BAR": "off",
    "TQDM_DISABLE": "1",
})

# ---------------------------------------------------------------------------
# Bash tool — same definition as mini-swe-agent v2
# ---------------------------------------------------------------------------

BASH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "required": ["command"],
        },
    },
}


@tool
def bash(command: str) -> str:
    """Execute a bash command."""
    # Placeholder — actual execution is done in bash_executor_node via Harbor
    return command


# ---------------------------------------------------------------------------
# Observation formatting — uses mini-swe-agent v2's Jinja2 template
# ---------------------------------------------------------------------------


def format_observation(output: str, returncode: int, exception_info: str = "") -> str:
    """Format command output using mini-swe-agent v2's observation template."""
    return Template(OBSERVATION_TEMPLATE, undefined=StrictUndefined).render(
        output={"output": output, "returncode": returncode, "exception_info": exception_info},
    )


# ---------------------------------------------------------------------------
# LangGraph nodes and edges
# ---------------------------------------------------------------------------


async def agent_node(state: MessagesState, config: RunnableConfig) -> dict:
    """Call the LLM with the bash tool bound."""
    model = config["configurable"]["model"]
    # Bind the bash tool so the LLM uses tool-calling format
    model_with_tools = model.bind_tools([bash])
    response = await model_with_tools.ainvoke(state["messages"])
    return {"messages": [response]}


async def bash_executor_node(state: MessagesState, config: RunnableConfig) -> dict:
    """Execute tool calls from the last AI message in the Harbor sandbox."""
    sandbox_manager = config["configurable"]["sandbox_manager"]
    sandbox = config["configurable"]["sandbox"]
    command_timeout = config["configurable"].get("command_timeout", 120)

    last_message = state["messages"][-1]
    assert isinstance(last_message, AIMessage)

    results = []
    submitted = False
    for tool_call in last_message.tool_calls:
        command = tool_call["args"].get("command", "")

        exec_result = await sandbox_manager.execute(sandbox, command, timeout=command_timeout)
        output = (exec_result.stdout or "") + ("\n" + exec_result.stderr if exec_result.stderr else "")

        # Check for submission in stdout (mini-swe-agent v2 style)
        if SUBMIT_SENTINEL in output and exec_result.return_code == 0:
            submitted = True

        observation = format_observation(
            output=output,
            returncode=exec_result.return_code,
        )

        results.append(
            ToolMessage(
                content=observation,
                tool_call_id=tool_call["id"],
                name="bash",
                additional_kwargs={"submitted": submitted},
            )
        )

    return {"messages": results}


def format_error_node(state: MessagesState, config: RunnableConfig) -> dict:
    """Send format error feedback when model doesn't produce tool calls."""
    return {"messages": [HumanMessage(content=FORMAT_ERROR_TEMPLATE)]}


def route_after_agent(state: MessagesState, config: RunnableConfig) -> Literal["bash_executor", "format_error", "__end__"]:
    """Route after the agent node."""
    max_turns = config["configurable"].get("max_turns", 30)

    num_agent_turns = sum(1 for m in state["messages"] if isinstance(m, AIMessage))
    if num_agent_turns > max_turns:
        logger.info(f"Turn limit exceeded ({num_agent_turns}/{max_turns})")
        return END

    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        return END

    # Check for submission in tool calls
    for tc in last_message.tool_calls:
        cmd = tc["args"].get("command", "")
        if SUBMIT_SENTINEL in cmd:
            logger.info("Agent submitted solution")
            return END

    # Check for submission in content (some models put it in text)
    content = last_message.content or ""
    if SUBMIT_SENTINEL in content:
        logger.info("Agent submitted solution (in content)")
        return END

    # Has tool calls → execute them
    if last_message.tool_calls:
        return "bash_executor"

    # No tool calls — retry with format error feedback (mini-swe-agent v2 behavior)
    format_errors = sum(
        1 for m in state["messages"]
        if isinstance(m, HumanMessage) and m.content == FORMAT_ERROR_TEMPLATE
    )
    if format_errors < MAX_FORMAT_RETRIES:
        logger.warning(f"No tool calls, sending format error feedback (retry {format_errors + 1}/{MAX_FORMAT_RETRIES})")
        return "format_error"

    logger.warning(f"No tool calls after {MAX_FORMAT_RETRIES} retries, ending")
    return END


def route_after_bash(state: MessagesState, config: RunnableConfig) -> Literal["agent", "__end__"]:
    """Route after bash executor — check if submission was detected in output."""
    for msg in reversed(state["messages"]):
        if not isinstance(msg, ToolMessage):
            break
        if msg.additional_kwargs.get("submitted"):
            logger.info("Agent submitted solution (detected in stdout)")
            return END
    return "agent"


def build_swe_agent_graph() -> StateGraph:
    """Build the LangGraph StateGraph for the mini SWE agent.

    Graph: agent → route → bash_executor → route_bash → agent (loop)
                        → format_error → agent (retry)
                        → END (submit / turn limit / retry limit)
    """
    workflow = StateGraph(MessagesState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("bash_executor", bash_executor_node)
    workflow.add_node("format_error", format_error_node)

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        route_after_agent,
        {"bash_executor": "bash_executor", "format_error": "format_error", END: END},
    )
    workflow.add_conditional_edges(
        "bash_executor",
        route_after_bash,
        {"agent": "agent", END: END},
    )
    workflow.add_edge("format_error", "agent")

    return workflow.compile()


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
        raise IndexError(f"Task index {task_index} out of range (has {len(downloaded_tasks)} tasks)")
    task_path = str(downloaded_tasks[task_index].local_path)
    return task_path, Task(task_path).instruction


def load_harbor_task_from_path(task_path: str) -> tuple[str, str]:
    """Load a single task from a local Harbor task directory."""
    from harbor import Task
    return task_path, Task(task_path).instruction


# ---------------------------------------------------------------------------
# Trajectory printing — full output, no truncation
# ---------------------------------------------------------------------------


def print_trajectory(messages):
    """Print the full agent trajectory."""
    turn = 0
    for msg in messages:
        if isinstance(msg, SystemMessage):
            print(f"\n{'='*60}")
            print(f"SYSTEM PROMPT ({len(msg.content)} chars)")
            print(f"{'='*60}")

        elif isinstance(msg, HumanMessage):
            print(f"\n{'='*60}")
            print(f"TASK ({len(msg.content)} chars)")
            print(f"{'='*60}")
            print(msg.content)

        elif isinstance(msg, AIMessage):
            turn += 1
            print(f"\n{'='*60}")
            print(f"TURN {turn}")
            print(f"{'='*60}")
            if msg.content:
                print(msg.content)
            for tc in msg.tool_calls:
                print(f"\n> COMMAND: {tc['args'].get('command', '')}")

        elif isinstance(msg, ToolMessage):
            print(f"\n--- Observation ---")
            print(msg.content)


# ---------------------------------------------------------------------------
# Main standalone execution
# ---------------------------------------------------------------------------


def render_template(template: str, **kwargs) -> str:
    """Render a Jinja2 template with the given variables."""
    return Template(template, undefined=StrictUndefined).render(**kwargs)


async def run_standalone(
    task_path: str,
    instruction: str,
    model_name: str = "gpt-4o",
    max_turns: int = 30,
    command_timeout: int = 120,
    env_type: str = "e2b",
    verbose: bool = True,
) -> tuple[list, float]:
    """Run the mini SWE agent on a single Harbor task.

    Returns:
        Tuple of (messages list, reward score).
    """
    from langchain_openai import ChatOpenAI
    from recipe.mini_swe_agent.harbor_sandbox import HarborSandboxManager

    # 1. Create LLM
    llm = ChatOpenAI(model=model_name, temperature=0.0)

    # 2. Create Harbor sandbox
    sandbox_manager = HarborSandboxManager(env_type=env_type)

    if verbose:
        print(f"Creating {env_type} sandbox for task: {task_path}")
    sandbox = await sandbox_manager.create(task_path)

    try:
        # 3. Build messages — render templates like mini-swe-agent does
        template_vars = {
            "task": instruction,
            "system": "Linux",
            "release": "",
            "version": "",
            "machine": "x86_64",
        }
        system_content = render_template(SYSTEM_TEMPLATE, **template_vars)
        instance_content = render_template(INSTANCE_TEMPLATE, **template_vars)

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=instance_content),
        ]

        # 4. Build and invoke LangGraph
        graph = build_swe_agent_graph()

        if verbose:
            print(f"Starting agent loop (max {max_turns} turns)...")

        config = {
            "configurable": {
                "model": llm,
                "sandbox_manager": sandbox_manager,
                "sandbox": sandbox,
                "max_turns": max_turns,
                "command_timeout": command_timeout,
            },
            "recursion_limit": max(50, max_turns * 3),
        }

        turn = 0
        final_messages = messages[:]

        async for event in graph.astream({"messages": messages}, config=config, stream_mode="updates"):
            for node_name, update in event.items():
                for msg in update.get("messages", []):
                    final_messages.append(msg)

                    if not verbose:
                        continue

                    if isinstance(msg, AIMessage):
                        turn += 1
                        print(f"\n{'='*60}")
                        print(f"TURN {turn}")
                        print(f"{'='*60}")
                        if msg.content:
                            print(msg.content)
                        for tc in msg.tool_calls:
                            print(f"\n> COMMAND: {tc['args'].get('command', '')}")
                        sys.stdout.flush()

                    elif isinstance(msg, ToolMessage):
                        print(f"\n--- Observation ---")
                        print(msg.content)
                        sys.stdout.flush()

                    elif isinstance(msg, HumanMessage) and msg.content == FORMAT_ERROR_TEMPLATE:
                        print(f"\n--- Format Error (retrying) ---")
                        print(msg.content[:200])
                        sys.stdout.flush()

        # 5. Compute reward
        if verbose:
            print("\nRunning verification tests...")
        reward = await sandbox_manager.compute_reward(sandbox)
        if verbose:
            print(f"Reward: {reward}")

        return final_messages, reward

    finally:
        if verbose:
            print("Destroying sandbox...")
        await sandbox_manager.destroy(sandbox)


def main():
    parser = argparse.ArgumentParser(
        description="Run mini SWE agent (LangGraph) on a Harbor task with E2B sandbox.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python recipe/mini_swe_agent/run_standalone.py \\
      --dataset swesmith@1.0 --task-index 0 --model gpt-4o

  python recipe/mini_swe_agent/run_standalone.py \\
      --task-path /path/to/task --model gpt-4o-mini
""",
    )

    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--dataset", type=str, help="Harbor dataset identifier")
    task_group.add_argument("--task-path", type=str, help="Path to a local Harbor task directory")

    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--command-timeout", type=int, default=120)
    parser.add_argument("--env-type", type=str, default="e2b", choices=["e2b", "docker", "daytona"])
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    log_level = logging.INFO if not args.quiet else logging.WARNING
    logging.basicConfig(level=log_level, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    for noisy in ["httpcore", "httpx", "urllib3", "openai", "e2b", "langsmith"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.dataset:
        task_path, instruction = load_harbor_task_from_dataset(args.dataset, args.task_index)
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
