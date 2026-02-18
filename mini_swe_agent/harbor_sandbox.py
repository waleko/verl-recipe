"""Harbor BaseEnvironment wrapper for verl training and standalone use.

Wraps Harbor's environment creation, command execution, verification,
and teardown into a simple interface for the mini SWE agent.

This module is generic — it works with any Harbor task that follows the
standard structure (instruction.md, task.toml, environment/, tests/).

Requires:
    pip install harbor
    E2B_API_KEY environment variable (for E2B backend)
"""

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

try:
    from harbor import ExecResult, Task
    from harbor.environments.factory import EnvironmentFactory
    from harbor.models.environment_type import EnvironmentType
    from harbor.models.trial.paths import TrialPaths
    from harbor.verifier.verifier import Verifier

    HARBOR_AVAILABLE = True
except ImportError:
    HARBOR_AVAILABLE = False

if TYPE_CHECKING:
    from harbor.environments.base import BaseEnvironment


ENV_TYPE_MAP = {
    "e2b": "E2B",
    "docker": "DOCKER",
    "daytona": "DAYTONA",
    "modal": "MODAL",
}


def _patch_e2b_sandbox_timeout(env: "BaseEnvironment", timeout: int) -> None:
    """Monkey-patch Harbor's E2B environment to use a custom sandbox timeout.

    Harbor hardcodes timeout=86400 (24h) in _create_sandbox, which exceeds
    E2B free tier's 1h limit. This patches the method to use our timeout.
    """
    from e2b import AsyncSandbox
    import types

    orig = env._create_sandbox

    async def _patched(self_env=env):
        metadata = {
            "environment_name": self_env.environment_name,
            "session_id": self_env.session_id,
        }
        self_env._sandbox = await AsyncSandbox.create(
            template=self_env._template_name,
            metadata=metadata,
            timeout=timeout,
            allow_internet_access=self_env.task_env_config.allow_internet,
        )

    env._create_sandbox = _patched


class HarborSandbox:
    """A single Harbor sandbox: environment + task + trial paths.

    Bundles everything needed to execute commands and verify results
    for one task instance.
    """

    def __init__(
        self,
        task: "Task",
        env: "BaseEnvironment",
        trial_paths: "TrialPaths",
    ):
        self.task = task
        self.env = env
        self.trial_paths = trial_paths


class HarborSandboxManager:
    """Creates and manages Harbor environments for agent execution.

    Supports E2B, Docker, Daytona, and Modal backends via Harbor's
    EnvironmentFactory. Rate limiting via asyncio.Semaphore controls
    concurrent sandbox count.
    """

    def __init__(
        self,
        max_concurrent: int = 10,
        default_timeout: int = 120,
        env_type: str = "e2b",
        trial_base_dir: str | None = None,
        sandbox_timeout: int | None = 3600,
    ):
        """Initialize the sandbox manager.

        Args:
            max_concurrent: Maximum number of concurrent sandboxes.
            default_timeout: Default command execution timeout in seconds.
            env_type: Harbor environment type ("e2b", "docker", "daytona", "modal").
            trial_base_dir: Base directory for trial output. Uses temp dir if None.
            sandbox_timeout: E2B sandbox lifetime in seconds. Default 3600 (1h)
                to stay within free tier. Set to None to use Harbor's default (24h).
        """
        if not HARBOR_AVAILABLE:
            raise ImportError(
                "Harbor is not installed. Install with: pip install harbor\n"
                "Also ensure E2B_API_KEY is set for E2B backend."
            )

        self._semaphore = asyncio.Semaphore(max_concurrent)
        self.default_timeout = default_timeout
        self.env_type_str = env_type
        self.env_type = EnvironmentType[ENV_TYPE_MAP.get(env_type, env_type.upper())]
        self.trial_base_dir = Path(trial_base_dir) if trial_base_dir else None
        self.sandbox_timeout = sandbox_timeout

    async def create(self, task_path: str) -> HarborSandbox:
        """Create and start a Harbor environment from a task directory.

        Args:
            task_path: Path to Harbor task directory (contains task.toml,
                instruction.md, environment/, tests/).

        Returns:
            HarborSandbox with started environment.
        """
        async with self._semaphore:
            task = Task(task_path)
            session_id = str(uuid4())

            # Create trial output directory
            if self.trial_base_dir:
                trial_dir = self.trial_base_dir / session_id
            else:
                trial_dir = Path(tempfile.mkdtemp(prefix=f"harbor_trial_{task.name}_"))

            trial_paths = TrialPaths(trial_dir=trial_dir)
            trial_paths.mkdir()

            env = EnvironmentFactory.create_environment(
                type=self.env_type,
                environment_dir=task.paths.environment_dir,
                environment_name=task.name,
                session_id=session_id,
                trial_paths=trial_paths,
                task_env_config=task.config.environment,
                logger=logger,
            )

            # Harbor hardcodes E2B sandbox timeout to 86400s (24h) which
            # exceeds the free tier limit of 3600s. Patch it.
            if self.env_type == EnvironmentType.E2B and self.sandbox_timeout is not None:
                _patch_e2b_sandbox_timeout(env, self.sandbox_timeout)

            await env.start(force_build=False)
            return HarborSandbox(task=task, env=env, trial_paths=trial_paths)

    async def execute(
        self,
        sandbox: HarborSandbox,
        command: str,
        timeout: Optional[int] = None,
    ) -> "ExecResult":
        """Execute a bash command in a Harbor environment.

        Args:
            sandbox: The HarborSandbox to execute in.
            command: Bash command string to execute.
            timeout: Command timeout in seconds (uses default if None).

        Returns:
            Harbor's ExecResult with stdout, stderr, and return_code.
        """
        timeout = timeout or self.default_timeout
        try:
            return await sandbox.env.exec(command, timeout_sec=timeout)
        except asyncio.TimeoutError:
            return ExecResult(
                stdout="",
                stderr=f"Command timed out after {timeout}s: {command[:200]}",
                return_code=-1,
            )
        except Exception as e:
            return ExecResult(
                stdout="",
                stderr=f"Execution error: {e}",
                return_code=-1,
            )

    async def compute_reward(self, sandbox: HarborSandbox) -> float:
        """Run verification tests and read the reward.

        Uses Harbor's Verifier class which:
        1. Uploads tests/ directory to the container
        2. Runs test.sh inside the container
        3. Downloads verifier logs (if environment is not mounted)
        4. Parses reward from reward.txt or reward.json

        Args:
            sandbox: The HarborSandbox to verify.

        Returns:
            Reward score (typically 0.0 or 1.0).
        """
        try:
            verifier = Verifier(
                task=sandbox.task,
                trial_paths=sandbox.trial_paths,
                environment=sandbox.env,
                logger=logger,
            )
            result = await verifier.verify()
            if result.rewards and "reward" in result.rewards:
                return float(result.rewards["reward"])
            elif result.rewards:
                # Take first reward value if key isn't "reward"
                return float(next(iter(result.rewards.values())))
            return 0.0
        except Exception as e:
            logger.warning(f"Verification failed: {e}")
            return 0.0

    async def destroy(self, sandbox: HarborSandbox) -> None:
        """Tear down a Harbor environment.

        Args:
            sandbox: The sandbox to destroy.
        """
        try:
            await sandbox.env.stop(delete=True)
        except Exception as e:
            logger.warning(f"Error destroying environment: {e}")


def load_task_instruction(task_path: str) -> str:
    """Load the instruction text from a Harbor task directory.

    Uses Harbor's Task class which reads instruction.md directly.

    Args:
        task_path: Path to Harbor task directory.

    Returns:
        The instruction text for the task.
    """
    if HARBOR_AVAILABLE:
        task = Task(task_path)
        return task.instruction

    # Fallback for when Harbor is not installed (e.g., testing)
    instruction_path = Path(task_path) / "instruction.md"
    if instruction_path.exists():
        return instruction_path.read_text()
    raise FileNotFoundError(f"No instruction.md found in {task_path}")
