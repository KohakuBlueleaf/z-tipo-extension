import sys
import pkg_resources

try:
    import launch
except:
    from tipo_installer import *

    install_llama_cpp()
    install_tipo_kgen()
else:

    from tipo_installer import logger, KGEN_VERSION

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

        launch.run_pip(
            f"install {package}",
            "LLaMA-CPP-Python for TIPO",
        )

    def install_llama_cpp():
        if get_installed_version("llama_cpp_python") is not None:
            return
        logger.info("Attempting to install LLaMA-CPP-Python")
        import torch

        has_cuda = torch.cuda.is_available()
        cuda_version = torch.version.cuda.replace(".", "") if has_cuda else ""
        arch = "cu" + cuda_version if has_cuda else "cpu"
        if has_cuda and arch >= "cu124":
            arch = "cu124"
        platform = sys.platform
        py_ver = f"cp{sys.version_info.major}{sys.version_info.minor}"
        if platform == "darwin":
            platform = "maxosx_11_0_arm64"
        elif platform == "win32":
            platform = "win_amd64"
        elif platform == "linux":
            platform = "linux_x86_64"

        for version, (py_vers, archs, platforms) in version_arch.items():
            if py_ver in py_vers and arch in archs and platform in platforms:
                break
        else:
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
            launch.run_pip(
                f"install {wheel}",
                "LLaMA-CPP-Python for TIPO",
            )
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
        launch.run_pip(
            f'install -U "tipo-kgen>={KGEN_VERSION}"',
            "tipo-kgen for TIPO",
        )

    install_llama_cpp()
    install_tipo_kgen()
