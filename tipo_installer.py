##
## ====================== Installer ======================
##
import copy
import logging
import os
import subprocess
import sys

import pkg_resources

KGEN_VERSION = "0.2.0"
python = sys.executable


def run(command) -> str:
    run_kwargs = {
        "args": command,
        "shell": True,
        "env": os.environ,
        "encoding": "utf8",
        "errors": "ignore",
    }
    run_kwargs["stdout"] = run_kwargs["stderr"] = subprocess.PIPE
    result = subprocess.run(**run_kwargs)

    if result.returncode != 0:
        raise RuntimeError()

    return result.stdout or ""


def run_pip(command):
    return run(f'"{python}" -m pip {command}')


class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[0;36m",  # CYAN
        "INFO": "\033[0;32m",  # GREEN
        "WARNING": "\033[0;33m",  # YELLOW
        "ERROR": "\033[0;31m",  # RED
        "CRITICAL": "\033[0;37;41m",  # WHITE ON RED
        "RESET": "\033[0m",  # RESET COLOR
    }

    def format(self, record):
        colored_record = copy.copy(record)
        levelname = colored_record.levelname
        seq = self.COLORS.get(levelname, self.COLORS["RESET"])
        colored_record.levelname = f"{seq}{levelname}{self.COLORS['RESET']}"
        return super().format(colored_record)


# Create a new logger
logger = logging.getLogger("TIPO-KGen-installer")
logger.propagate = False

# Add handler if we don't have one.
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        ColoredFormatter(
            "[%(name)s]-|%(asctime)s|-%(levelname)s: %(message)s", "%H:%M:%S"
        )
    )
    logger.addHandler(handler)

logger.setLevel(logging.INFO)
logger.debug("Logger initialized.")


def get_installed_version(package: str):
    try:
        return pkg_resources.get_distribution(package).version
    except Exception:
        return None


llama_cpp_python_wheel = (
    "llama-cpp-python --prefer-binary "
    "--extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/{}/{}"
)
llama_cpp_python_wheel_official = (
    "https://github.com/abetlen/llama-cpp-python/releases/download/"
    "v{version_arch}/llama_cpp_python-{version}-{python}-{python}-{platform}.whl"
)
version_arch = {
    "0.3.4": [
        ("cp39", "cp310", "cp311", "cp312"),
        ("cu121", "cu122", "cu123", "cu124", "metal"),
        ("linux_x86_64", "win_amd64", "maxosx_11_0_arm64"),
    ],
    "0.3.2": [
        ("cp38", "cp39", "cp310", "cp311", "cp312"),
        ("cpu", "metal"),
        ("linux_x86_64", "win_amd64", "maxosx_11_0_arm64"),
    ],
}


def install_llama_cpp_legacy(cuda_version, has_cuda):
    if cuda_version >= "122":
        cuda_version = "122"
    package = llama_cpp_python_wheel.format(
        "AVX2", f"cu{cuda_version}" if has_cuda else "cpu"
    )

    run_pip(f"install {package}")


def install_llama_cpp():
    if get_installed_version("llama_cpp_python") is not None:
        return
    logger.info("Attempting to install LLaMA-CPP-Python")
    import torch

    platform = sys.platform
    py_ver = f"cp{sys.version_info.major}{sys.version_info.minor}"
    if platform == "darwin":
        platform = "maxosx_11_0_arm64"
    elif platform == "win32":
        platform = "win_amd64"
    elif platform == "linux":
        platform = "linux_x86_64"

    has_cuda = torch.cuda.is_available()
    has_metal = torch.backends.mps.is_available() and torch.backends.mps.is_built()

    if has_cuda:
        cuda_version = torch.version.cuda.replace(".", "")
        arch = "cu" + cuda_version
    elif has_metal:
        # torch.version.cuda is None on Apple Silicon
        cuda_version = "metal"
        arch = "metal"
    else:
        cuda_version = ""
        arch = "cpu"

    if has_cuda and arch > "cu124":
        arch = "cu124"

    for version, (py_vers, archs, platforms) in version_arch.items():
        if py_ver in py_vers and arch in archs and platform in platforms:
            break
    else:
        if cuda_version == "metal":
            logger.warning(
                "Metal Performance Shaders detected. "
                "Prebuilt llama-cpp-python may not be available. "
                "For better performance, you may need to reinstall and build it manually. "
                "Goto: https://github.com/abetlen/llama-cpp-python/"
            )

        logger.warning("Official wheel not found, using legacy builds")
        install_llama_cpp_legacy(cuda_version, has_cuda)
        return

    wheel = llama_cpp_python_wheel_official.format(
        version=version,
        python=py_ver,
        platform=platform,
        version_arch=f"{version}-{arch}",
    )

    try:
        run_pip(f"install {wheel}")
        logger.info("Installation of llama-cpp-python succeeded")
    except Exception:
        logger.warning(
            "Installation of llama-cpp-python failed, "
            "Please try to install it manually or use non-gguf models"
        )


def install_tipo_kgen():
    version = get_installed_version("tipo-kgen")
    if version is not None and version >= KGEN_VERSION:
        return
    logger.info("Attempting to install tipo_kgen")
    run_pip(f'install -U "tipo-kgen>={KGEN_VERSION}"')
