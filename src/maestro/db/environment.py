"""
MAESTRO — Runtime environment capture for reproducibility.

Captures a snapshot of OS, hardware, Python runtime, git state, key library
versions and (optionally) the Docker image digest once per CLI invocation.
The resulting ``RunEnvironment`` is persisted to the ``run_environments``
table and its ``environment_id`` is propagated into every ``RunConfig`` that
shares the invocation, so any future replication attempt can diagnose
diverging numbers against the exact stack that produced the original data.

Failure modes are intentionally soft: if git is unavailable, the working
tree is detached, ``MAESTRO_IMAGE_DIGEST`` is unset, or a library is not
installed, the field is recorded as ``None`` and the experiment continues.
The whole point of this module is observability — it must never abort the
run it is supposed to describe.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

from maestro.schemas import RunEnvironment

# Libraries whose installed version materially changes experiment behavior.
# Mirrors the runtime dependencies in pyproject.toml — pure dev tooling
# (pytest, ruff) is intentionally excluded since it has no effect on the
# numbers produced by a run.
#
# KEEP IN SYNC WITH pyproject.toml [project].dependencies. Adding a runtime
# dep there and forgetting to add it here means its version silently stops
# being recorded in run_environments.lib_versions — and the omission only
# surfaces when a future replication attempt diverges and you wonder why
# the env snapshot looks "complete" but is missing the smoking gun.
_LIB_WHITELIST: tuple[str, ...] = (
    "anthropic",
    "openai",
    "mistralai",
    "google-genai",
    "crewai",
    "langgraph",
    "pydantic",
    "python-dotenv",
    # Analysis pipeline deps — their version changes the statistics output
    # (ANOVA implementation details, default ddof, etc.), so capture them.
    "statsmodels",
    "pandas",
    # Windows-only via env marker in pyproject.toml. On Linux/macOS this
    # will record as None (not installed), which is correct — the system
    # zoneinfo DB is in use there. On Windows, recording the tzdata wheel
    # version closes the reproducibility loop for timestamp display.
    "tzdata",
)


def _git_head() -> str | None:
    """Return ``git rev-parse HEAD`` or ``None`` if git is unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        # OSError covers FileNotFoundError (git not on PATH), PermissionError
        # (git present but not executable) and other low-level subprocess
        # spawn failures. The probe must never crash the experiment runner
        # it is supposed to describe — fall back to None.
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _git_dirty() -> bool | None:
    """
    Return True if the working tree has uncommitted changes, False if clean,
    None if git status could not be determined. Tri-state on purpose: a
    failed probe must not be silently equated with a clean tree, which
    would produce a falsely reassuring reproducibility record.
    """
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        # OSError covers FileNotFoundError (git not on PATH), PermissionError
        # (git present but not executable) and other low-level subprocess
        # spawn failures. The probe must never crash the experiment runner
        # it is supposed to describe — fall back to None.
        return None
    if out.returncode != 0:
        return None
    return bool(out.stdout.strip())


def _lib_versions() -> dict[str, str | None]:
    """Resolve installed versions for every whitelisted library, ``None`` if absent."""
    resolved: dict[str, str | None] = {}
    for name in _LIB_WHITELIST:
        try:
            resolved[name] = version(name)
        except PackageNotFoundError:
            resolved[name] = None
    return resolved


def capture_environment(
    image_digest_env: str = "MAESTRO_IMAGE_DIGEST",
) -> RunEnvironment:
    """
    Snapshot the runtime environment for this CLI invocation.

    ``image_digest_env`` is parameterised so tests can pass an alternative
    variable name without mutating ``os.environ``. The default matches the
    convention documented in the issue.
    """
    import os

    return RunEnvironment(
        os=platform.platform(),
        arch=platform.machine(),
        python=sys.version,
        hostname=platform.node() or None,
        git_commit=_git_head(),
        git_dirty=_git_dirty(),
        lib_versions=json.dumps(_lib_versions(), sort_keys=True),
        docker_image_digest=os.environ.get(image_digest_env) or None,
        captured_at=datetime.now(timezone.utc),
    )
