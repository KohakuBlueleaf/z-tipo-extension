import launch
import pkg_resources


def get_installed_version(package: str):
    try:
        return pkg_resources.get_distribution(package).version
    except Exception:
        return None


llama_cpp_python_wheel = (
    "llama-cpp-python --prefer-binary "
    "--extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/{}/{}"
)


def install_llama_cpp():
    if get_installed_version("llama_cpp_python") is not None:
        return
    print("Attempting to install LLaMA-CPP-Python")
    import torch

    has_cuda = torch.cuda.is_available()
    cuda_version = torch.version.cuda.replace(".", "")
    package = llama_cpp_python_wheel.format(
        "AVX2", f"cu{cuda_version}" if has_cuda else "cpu"
    )

    launch.run_pip(
        f"install {package}",
        f"LLaMA-CPP-Python for DanTagGen",
    )


def install_tipo_kgen():
    version = get_installed_version("tipo-kgen")
    if version is not None and version >= "0.0.3":
        return
    print("Attempting to install tipo_kgen")
    launch.run_pip(
        f"install -U tipo-kgen>=0.0.3",
        f"tipo-kgen for DanTagGen",
    )


install_llama_cpp()
install_tipo_kgen()
