"""
aria/capabilities/external/system_info.py — System information capabilities.

Provides read-only access to:
    - OS / hardware / disk / memory statistics
    - Current date/time
    - Environment variables (with secret filtering)

Example:
    >>> from aria.capabilities.registry import GLOBAL_REGISTRY
    >>> r = await GLOBAL_REGISTRY.execute("get_system_info", {})
    >>> print(r.data["os"])
    'Darwin'
"""

from __future__ import annotations

import datetime
import getpass
import os
import platform
import sys
from typing import Optional

import psutil
from pydantic import Field

from aria.capabilities.base import (
    Capability,
    CapabilityInput,
    CapabilityOutput,
)
from aria.core.logger import get_logger

log = get_logger("capabilities.system_info")

# Keys containing any of these substrings should never be returned.
_SECRET_SUBSTRINGS = {"SECRET", "KEY", "PASSWORD", "TOKEN"}


def _is_secret_key(key: str) -> bool:
    """Check whether an env-var key looks like a secret.

    Args:
        key: The environment variable name.

    Returns:
        True if the key contains a secret-looking substring.
    """
    upper = key.upper()
    return any(s in upper for s in _SECRET_SUBSTRINGS)


# ═══════════════════════════════════════════════════════════════════════
# get_system_info
# ═══════════════════════════════════════════════════════════════════════


class GetSystemInfoInput(CapabilityInput):
    """Input schema for ``get_system_info`` (no parameters required)."""

    pass


class GetSystemInfoCapability(Capability):
    """Return a snapshot of system information.

    Includes OS, hostname, CPU, memory, disk, Python version, and the
    current user.

    Example input::

        {}

    Example output::

        {
            "os": "Darwin",
            "hostname": "Nitish-MacBook.local",
            "cpu_count": 8,
            "memory_total_gb": 16.0,
            "memory_available_gb": 8.3,
            "disk_total_gb": 500.0,
            "disk_free_gb": 120.5,
            "python_version": "3.9.6",
            "current_user": "nitish"
        }
    """

    name = "get_system_info"
    description = "Get OS, CPU, memory, disk, and Python version information."
    input_schema = GetSystemInfoInput
    tags = ["external", "system", "read"]
    requires_confirmation = False

    async def execute(
        self, input_data: GetSystemInfoInput
    ) -> CapabilityOutput:
        """Collect system info.

        Args:
            input_data: Empty validated input.

        Returns:
            CapabilityOutput with system details dict.
        """
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        return CapabilityOutput(
            success=True,
            data={
                "os": platform.system(),
                "hostname": platform.node(),
                "cpu_count": os.cpu_count() or 0,
                "memory_total_gb": round(mem.total / (1024 ** 3), 1),
                "memory_available_gb": round(mem.available / (1024 ** 3), 1),
                "disk_total_gb": round(disk.total / (1024 ** 3), 1),
                "disk_free_gb": round(disk.free / (1024 ** 3), 1),
                "python_version": platform.python_version(),
                "current_user": getpass.getuser(),
            },
        )


# ═══════════════════════════════════════════════════════════════════════
# get_current_time
# ═══════════════════════════════════════════════════════════════════════


class GetCurrentTimeInput(CapabilityInput):
    """Input schema for the ``get_current_time`` capability.

    Attributes:
        timezone: Timezone name.  Use ``"local"`` for the system timezone
            or an IANA name like ``"UTC"`` or ``"America/New_York"``.
    """

    timezone: str = Field(
        default="local",
        description="Timezone: 'local' or an IANA name like 'UTC'.",
    )


class GetCurrentTimeCapability(Capability):
    """Return the current date and time.

    Example input::

        {"timezone": "local"}

    Example output::

        {
            "iso": "2025-06-15T14:30:00.123456",
            "unix": 1750000200,
            "timezone": "local",
            "formatted": "Sun Jun 15 14:30:00 2025"
        }
    """

    name = "get_current_time"
    description = "Get the current date and time in ISO, Unix, and formatted forms."
    input_schema = GetCurrentTimeInput
    tags = ["external", "time", "read"]
    requires_confirmation = False

    async def execute(
        self, input_data: GetCurrentTimeInput
    ) -> CapabilityOutput:
        """Get current time.

        Args:
            input_data: Validated ``GetCurrentTimeInput``.

        Returns:
            CapabilityOutput with iso, unix, timezone, formatted.
        """
        tz_name = input_data.timezone

        if tz_name.lower() == "local":
            now = datetime.datetime.now()
        elif tz_name.upper() == "UTC":
            now = datetime.datetime.utcnow()
        else:
            # Try with zoneinfo (stdlib in 3.9+)
            try:
                from zoneinfo import ZoneInfo

                tz = ZoneInfo(tz_name)
                now = datetime.datetime.now(tz)
            except (ImportError, KeyError):
                return CapabilityOutput(
                    success=False,
                    error=f"Unknown timezone: {tz_name}",
                )

        return CapabilityOutput(
            success=True,
            data={
                "iso": now.isoformat(),
                "unix": int(now.timestamp()),
                "timezone": tz_name,
                "formatted": now.strftime("%a %b %d %H:%M:%S %Y"),
            },
        )


# ═══════════════════════════════════════════════════════════════════════
# get_env_var
# ═══════════════════════════════════════════════════════════════════════


class GetEnvVarInput(CapabilityInput):
    """Input schema for the ``get_env_var`` capability.

    Attributes:
        key: The environment variable name.
        default: Value to return if the variable is not set.
    """

    key: str = Field(description="Environment variable name.")
    default: Optional[str] = Field(
        default=None, description="Fallback value if not set."
    )


class GetEnvVarCapability(Capability):
    """Read a single environment variable's value.

    For security, variables whose names contain SECRET, KEY, PASSWORD,
    or TOKEN are never returned — the output will say
    ``"[REDACTED]"``.

    Example input::

        {"key": "HOME"}

    Example output::

        {"key": "HOME", "value": "/Users/nitish", "exists": true}
    """

    name = "get_env_var"
    description = "Read an environment variable (secrets are redacted)."
    input_schema = GetEnvVarInput
    tags = ["external", "env", "read"]
    requires_confirmation = False

    async def execute(self, input_data: GetEnvVarInput) -> CapabilityOutput:
        """Read an env var.

        Args:
            input_data: Validated ``GetEnvVarInput``.

        Returns:
            CapabilityOutput with key, value, and exists flag.
        """
        if _is_secret_key(input_data.key):
            return CapabilityOutput(
                success=True,
                data={
                    "key": input_data.key,
                    "value": "[REDACTED]",
                    "exists": input_data.key in os.environ,
                },
            )

        value = os.environ.get(input_data.key, input_data.default)
        return CapabilityOutput(
            success=True,
            data={
                "key": input_data.key,
                "value": value,
                "exists": input_data.key in os.environ,
            },
        )


# ═══════════════════════════════════════════════════════════════════════
# Capability instances for registration
# ═══════════════════════════════════════════════════════════════════════

get_system_info = GetSystemInfoCapability()
get_current_time = GetCurrentTimeCapability()
get_env_var = GetEnvVarCapability()

ALL_SYSTEM_INFO_CAPABILITIES = [
    get_system_info,
    get_current_time,
    get_env_var,
]
