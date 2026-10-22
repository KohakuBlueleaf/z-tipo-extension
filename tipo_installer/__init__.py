"""Dependency bootstrap shared by the ComfyUI nodes and the A1111 script.

Every entry point is idempotent and safe to call on each node execution: once a
dependency resolves the result is memoised, so the steady-state cost is a dict
lookup rather than a subprocess.
"""

import importlib
import importlib.metadata
import importlib.util
import os
import re
import sys

from . import hardware, pkg, scheme, wheels
from .log import logger

__all__ = [
    "ensure_llama_cpp",
    "ensure_runtime",
    "ensure_tipo_kgen",
    "install_llama_cpp",
    "install_tipo_kgen",
    "logger",
    "prepare_dll_path",
    "rebind_kgen_llama",
]

_llama_state = None
_kgen_state = None
_dll_ready = False


def _disabled():
    return os.environ.get(scheme.ENV_DISABLE, "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _importable(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def installed_version(distribution):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _as_tuple(text):
    return tuple(int(part) for part in re.findall(r"\d+", str(text or "")))


def prepare_dll_path():
    """Let llama.cpp's accelerator DLLs resolve against the ones torch ships."""
    global _dll_ready
    if _dll_ready or os.name != "nt":
        return
    _dll_ready = True
    torch = sys.modules.get("torch")
    if torch is None:
        return
    lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.isdir(lib):
        try:
            os.add_dll_directory(lib)
        except OSError:
            pass


def ensure_llama_cpp():
    """Install a prebuilt llama-cpp-python matching this GPU. True if importable."""
    global _llama_state
    if _llama_state is not None:
        return _llama_state

    if _importable(scheme.IMPORT_NAME):
        _llama_state = True
        return True

    if _disabled():
        logger.info(f"{scheme.ENV_DISABLE} is set; skipping llama-cpp-python install")
        _llama_state = False
        return False

    accelerator = hardware.detect()
    logger.info(f"Detected accelerator: {accelerator.describe()} on {pkg.host()}")

    candidate = wheels.resolve(accelerator)
    if candidate is None:
        logger.warning(
            "No prebuilt wheel matched this platform; trying a plain install"
        )
        ok = pkg.install([scheme.DISTRIBUTION], scheme.DISTRIBUTION)
    else:
        logger.info(f"Selected llama-cpp-python {candidate.describe()}")
        ok = pkg.install([candidate.url], f"llama-cpp-python {candidate.describe()}")

    importlib.invalidate_caches()
    _llama_state = ok and _importable(scheme.IMPORT_NAME)
    if not _llama_state:
        logger.warning(
            "llama-cpp-python is unavailable, so GGUF models cannot load. "
            "Non-GGUF TIPO models still work."
        )
    return _llama_state


def ensure_tipo_kgen():
    """Install or upgrade tipo-kgen to the minimum this extension needs."""
    global _kgen_state
    if _kgen_state is not None:
        return _kgen_state

    current = installed_version(scheme.KGEN_DISTRIBUTION)
    wanted = _as_tuple(scheme.KGEN_MIN_VERSION)
    if current is not None and _as_tuple(current) >= wanted:
        _kgen_state = True
        return True

    if _disabled():
        _kgen_state = _importable(scheme.KGEN_IMPORT_NAME)
        return _kgen_state

    requirement = f"{scheme.KGEN_DISTRIBUTION}>={scheme.KGEN_MIN_VERSION}"
    ok = pkg.install(["-U", requirement], requirement)
    importlib.invalidate_caches()
    _kgen_state = ok and _importable(scheme.KGEN_IMPORT_NAME)
    return _kgen_state


def rebind_kgen_llama():
    """Update KGen's loader and generation backend after a late installation."""
    try:
        from llama_cpp import Llama
    except ImportError:
        return
    for module_name in ("kgen.models", "kgen.generate"):
        module = sys.modules.get(module_name)
        if module is not None and module.Llama is not Llama:
            module.Llama = Llama
            logger.info(f"Rebound llama_cpp.Llama into the already-imported {module_name}")


def ensure_runtime():
    """Full bootstrap for a node execution; returns True if GGUF models can load."""
    ensure_tipo_kgen()
    prepare_dll_path()
    ready = ensure_llama_cpp()
    if ready:
        rebind_kgen_llama()
    return ready


install_llama_cpp = ensure_llama_cpp
install_tipo_kgen = ensure_tipo_kgen
