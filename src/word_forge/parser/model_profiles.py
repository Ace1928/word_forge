"""Resource-aware local language-model profiles for Word Forge.

The lexical pipeline is useful without a generative model. This module keeps
that offline path explicit while offering named, reproducible model choices for
systems ranging from constrained CPUs to accelerated workstations.
"""

from __future__ import annotations

import importlib.util
import os
import re
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Dict, Iterable, Literal, Optional, Tuple

Accelerator = Literal["cpu", "cuda", "mps"]


class ModelProfileError(ValueError):
    """Raised when a model profile name or runtime selection is invalid."""


@dataclass(frozen=True, slots=True)
class RuntimeResources:
    """Runtime capacity used for deterministic model recommendations."""

    total_ram_gib: float
    available_ram_gib: float
    cpu_threads: int
    accelerator: Accelerator = "cpu"
    accelerator_memory_gib: Optional[float] = None
    torch_version: Optional[str] = None
    transformers_version: Optional[str] = None

    @property
    def llm_dependencies_available(self) -> bool:
        """Return whether the local Transformers inference stack is installed."""
        return self.torch_version is not None and self.transformers_version is not None

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable runtime description."""
        return {
            "total_ram_gib": round(self.total_ram_gib, 2),
            "available_ram_gib": round(self.available_ram_gib, 2),
            "cpu_threads": self.cpu_threads,
            "accelerator": self.accelerator,
            "accelerator_memory_gib": (
                round(self.accelerator_memory_gib, 2)
                if self.accelerator_memory_gib is not None
                else None
            ),
            "torch_version": self.torch_version,
            "transformers_version": self.transformers_version,
        }


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """A documented model choice with conservative operational constraints."""

    name: str
    display_name: str
    model_id: Optional[str]
    description: str
    parameter_count_billions: float
    context_tokens: int
    minimum_available_ram_gib: float
    recommended_available_ram_gib: float
    minimum_transformers_version: Optional[str]
    license_name: str
    gated: bool = False
    prefers_accelerator: bool = False

    @property
    def enabled(self) -> bool:
        """Return whether this profile loads a generative model."""
        return self.model_id is not None

    def readiness(self, resources: RuntimeResources) -> Tuple[bool, Tuple[str, ...]]:
        """Evaluate whether this profile can be loaded by the current runtime."""
        if not self.enabled:
            return True, ()

        issues = []
        if not resources.llm_dependencies_available:
            issues.append("Install the 'llm' extra (word_forge[llm]).")
        elif self.minimum_transformers_version and not version_at_least(
            resources.transformers_version, self.minimum_transformers_version
        ):
            issues.append(
                "Transformers "
                f">={self.minimum_transformers_version} is required; found "
                f"{resources.transformers_version}."
            )
        if resources.available_ram_gib < self.minimum_available_ram_gib:
            issues.append(
                f"At least {self.minimum_available_ram_gib:g} GiB available RAM is "
                f"required; found {resources.available_ram_gib:.1f} GiB."
            )
        return not issues, tuple(issues)

    def warnings(self, resources: RuntimeResources) -> Tuple[str, ...]:
        """Return non-blocking operational cautions for this profile."""
        warnings = []
        if self.gated:
            warnings.append("Model access requires accepting its provider terms.")
        if self.prefers_accelerator and resources.accelerator == "cpu":
            warnings.append(
                "CPU inference is supported but may be slow; use quantization."
            )
        if (
            self.enabled
            and resources.available_ram_gib < self.recommended_available_ram_gib
            and resources.available_ram_gib >= self.minimum_available_ram_gib
        ):
            warnings.append(
                f"{self.recommended_available_ram_gib:g} GiB available RAM is recommended."
            )
        return tuple(warnings)

    def to_dict(
        self, resources: Optional[RuntimeResources] = None
    ) -> Dict[str, object]:
        """Return profile metadata and, optionally, runtime readiness."""
        result: Dict[str, object] = {
            "name": self.name,
            "display_name": self.display_name,
            "model_id": self.model_id,
            "description": self.description,
            "parameter_count_billions": self.parameter_count_billions,
            "context_tokens": self.context_tokens,
            "minimum_available_ram_gib": self.minimum_available_ram_gib,
            "recommended_available_ram_gib": self.recommended_available_ram_gib,
            "minimum_transformers_version": self.minimum_transformers_version,
            "license": self.license_name,
            "gated": self.gated,
            "prefers_accelerator": self.prefers_accelerator,
        }
        if resources is not None:
            ready, issues = self.readiness(resources)
            result.update(
                {
                    "ready": ready,
                    "issues": list(issues),
                    "warnings": list(self.warnings(resources)),
                }
            )
        return result


MODEL_PROFILES: Dict[str, ModelProfile] = {
    "off": ModelProfile(
        name="off",
        display_name="Offline lexical core",
        model_id=None,
        description="Use authoritative lexical sources without generative enrichment.",
        parameter_count_billions=0.0,
        context_tokens=0,
        minimum_available_ram_gib=0.0,
        recommended_available_ram_gib=0.0,
        minimum_transformers_version=None,
        license_name="n/a",
    ),
    "portable": ModelProfile(
        name="portable",
        display_name="Qwen 2.5 0.5B Instruct",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        description="Ungated Apache-2.0 text model suitable for CPU-first enrichment.",
        parameter_count_billions=0.49,
        context_tokens=32_768,
        minimum_available_ram_gib=2.5,
        recommended_available_ram_gib=4.0,
        minimum_transformers_version="4.37.0",
        license_name="Apache-2.0",
    ),
    "gemma3-tiny": ModelProfile(
        name="gemma3-tiny",
        display_name="Gemma 3 270M Instruct",
        model_id="google/gemma-3-270m-it",
        description="Smallest Gemma profile for constrained and on-device systems.",
        parameter_count_billions=0.27,
        context_tokens=32_768,
        minimum_available_ram_gib=2.0,
        recommended_available_ram_gib=3.0,
        minimum_transformers_version="4.53.0",
        license_name="Gemma Terms of Use",
        gated=True,
    ),
    "gemma4-edge": ModelProfile(
        name="gemma4-edge",
        display_name="Gemma 4 E2B Instruct",
        model_id="google/gemma-4-E2B-it",
        description="Multilingual Gemma 4 edge model used for text enrichment.",
        parameter_count_billions=5.1,
        context_tokens=131_072,
        minimum_available_ram_gib=14.0,
        recommended_available_ram_gib=20.0,
        minimum_transformers_version="5.5.0",
        license_name="Apache-2.0",
        prefers_accelerator=True,
    ),
}

DEFAULT_MODEL_PROFILE = "portable"
_default_model_id = MODEL_PROFILES[DEFAULT_MODEL_PROFILE].model_id
if _default_model_id is None:  # pragma: no cover - catalog invariant
    raise RuntimeError("The default model profile must define a model identifier")
DEFAULT_MODEL_ID: str = _default_model_id


def iter_model_profiles() -> Iterable[ModelProfile]:
    """Yield profiles in stable presentation order."""
    return MODEL_PROFILES.values()


def get_model_profile(name: str) -> ModelProfile:
    """Resolve a profile name using case-insensitive dash normalization."""
    normalized = name.strip().casefold().replace("_", "-")
    try:
        return MODEL_PROFILES[normalized]
    except KeyError as exc:
        available = ", ".join((*MODEL_PROFILES, "auto"))
        raise ModelProfileError(
            f"Unknown model profile '{name}'. Available profiles: {available}"
        ) from exc


def recommend_model_profile(resources: RuntimeResources) -> ModelProfile:
    """Recommend the fastest broadly accessible model that fits the runtime."""
    if not resources.llm_dependencies_available:
        return MODEL_PROFILES["off"]
    portable = MODEL_PROFILES["portable"]
    portable_ready, _ = portable.readiness(resources)
    if not portable_ready:
        return MODEL_PROFILES["off"]

    gemma4 = MODEL_PROFILES["gemma4-edge"]
    gemma4_ready, _ = gemma4.readiness(resources)
    accelerator_memory = resources.accelerator_memory_gib or 0.0
    if gemma4_ready and resources.accelerator == "cuda" and accelerator_memory >= 12.0:
        return gemma4
    return portable


def resolve_model_profile(
    name: str,
    resources: Optional[RuntimeResources] = None,
    *,
    require_ready: bool = False,
) -> ModelProfile:
    """Resolve a named or automatic model profile.

    Args:
        name: Profile name or ``auto``.
        resources: Optional deterministic runtime snapshot.
        require_ready: Raise :class:`ModelProfileError` when the selected
            profile cannot run with the detected dependencies and memory.

    Returns:
        The selected model profile.

    Raises:
        ModelProfileError: If the name is unknown or readiness is required but
            the runtime does not meet the profile's hard constraints.
    """
    resolved_resources = resources
    if name.strip().casefold() == "auto":
        resolved_resources = resources or detect_runtime_resources()
        profile = recommend_model_profile(resolved_resources)
    else:
        profile = get_model_profile(name)

    if require_ready and profile.enabled:
        resolved_resources = resolved_resources or detect_runtime_resources()
        ready, issues = profile.readiness(resolved_resources)
        if not ready:
            detail = " ".join(issues)
            raise ModelProfileError(
                f"Model profile '{profile.name}' is not ready: {detail} "
                "Run 'word_forge models list' for runtime details."
            )
    return profile


def detect_runtime_resources() -> RuntimeResources:
    """Inspect memory, dependency versions, and available accelerators safely."""
    total_ram, available_ram = _detect_memory_gib()
    accelerator: Accelerator = "cpu"
    accelerator_memory: Optional[float] = None
    torch_version = _package_version("torch")
    transformers_version = _package_version("transformers")

    if torch_version is not None and importlib.util.find_spec("torch") is not None:
        try:
            import torch

            if torch.cuda.is_available():
                accelerator = "cuda"
                properties = torch.cuda.get_device_properties(0)
                accelerator_memory = float(properties.total_memory) / (1024**3)
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                accelerator = "mps"
        except Exception:
            # Model listing and the lexical core must survive a broken optional
            # accelerator installation.
            accelerator = "cpu"
            accelerator_memory = None

    return RuntimeResources(
        total_ram_gib=total_ram,
        available_ram_gib=available_ram,
        cpu_threads=max(1, os.cpu_count() or 1),
        accelerator=accelerator,
        accelerator_memory_gib=accelerator_memory,
        torch_version=torch_version,
        transformers_version=transformers_version,
    )


def version_at_least(installed: Optional[str], required: str) -> bool:
    """Compare release-number components without another runtime dependency."""
    if installed is None:
        return False

    def release_parts(value: str) -> Tuple[int, ...]:
        match = re.match(r"\s*(\d+(?:\.\d+)*)", value)
        return tuple(int(part) for part in match.group(1).split(".")) if match else ()

    installed_parts = release_parts(installed)
    required_parts = release_parts(required)
    width = max(len(installed_parts), len(required_parts))
    return installed_parts + (0,) * (width - len(installed_parts)) >= (
        required_parts + (0,) * (width - len(required_parts))
    )


def _package_version(package_name: str) -> Optional[str]:
    """Return an installed distribution version without importing the package."""
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def _detect_memory_gib() -> Tuple[float, float]:
    """Return total and currently available memory with portable fallbacks."""
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        try:
            values = {}
            for line in meminfo.read_text(encoding="utf-8").splitlines():
                key, raw_value = line.split(":", 1)
                values[key] = int(raw_value.strip().split()[0]) * 1024
            total = values["MemTotal"]
            available = values.get("MemAvailable", values.get("MemFree", total))
            return total / (1024**3), available / (1024**3)
        except (KeyError, OSError, ValueError):
            pass

    if os.name == "nt":
        windows_memory = _detect_windows_memory_gib()
        if windows_memory is not None:
            return windows_memory

    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        total = page_size * int(os.sysconf("SC_PHYS_PAGES"))
        try:
            available = page_size * int(os.sysconf("SC_AVPHYS_PAGES"))
        except (KeyError, OSError, ValueError):
            available = total
        return total / (1024**3), available / (1024**3)
    except (AttributeError, KeyError, OSError, ValueError):
        # Unknown capacity is represented conservatively; automatic selection
        # will keep generative enrichment disabled.
        return 0.0, 0.0


def _detect_windows_memory_gib() -> Optional[Tuple[float, float]]:
    """Return Windows physical memory using the native kernel API."""
    try:
        import ctypes

        class MemoryStatus(ctypes.Structure):
            """Windows ``MEMORYSTATUSEX`` structure."""

            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        windows_libraries = getattr(ctypes, "windll", None)
        if windows_libraries is None:
            raise OSError("Windows native libraries are unavailable")
        if not windows_libraries.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        return (
            float(status.total_physical) / (1024**3),
            float(status.available_physical) / (1024**3),
        )
    except (AttributeError, OSError, ValueError):
        return None


__all__ = [
    "DEFAULT_MODEL_ID",
    "DEFAULT_MODEL_PROFILE",
    "MODEL_PROFILES",
    "ModelProfile",
    "ModelProfileError",
    "RuntimeResources",
    "detect_runtime_resources",
    "get_model_profile",
    "iter_model_profiles",
    "recommend_model_profile",
    "resolve_model_profile",
    "version_at_least",
]
