"""环境检查不依赖 GPU 或模型，也可以在缺少第三方依赖时运行。"""
import hashlib
import importlib.metadata
import platform
from pathlib import Path
import subprocess
import sys


def environment():
    # 检查 Python 版本、平台、依赖包版本、CUDA 可用性、GPU 信息等
    versions = {}
    for name in ("torch", "transformers", "numpy", "matplotlib", "PyYAML", "tokenizers", "pytest"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = {"python": sys.version, "platform": platform.platform(), "dependencies": versions,
              "cuda_available": False, "gpus": [], "deterministic_algorithms": False,
              "tf32": False, "compile": False, "cuda_graphs": False}
    if versions["torch"]:
        import torch
        result.update(cuda_available=torch.cuda.is_available(), torch_cuda_version=torch.version.cuda,
                      cpu_threads=torch.get_num_threads())
        if torch.cuda.is_available():
            result["gpus"] = [{"index": i, "name": torch.cuda.get_device_name(i),
                "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory,
                "capability": list(torch.cuda.get_device_capability(i))} for i in range(torch.cuda.device_count())]
        try:
            result["nvidia_smi"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            result["nvidia_smi"] = None
    return result


def code_identity():
    # 计算源代码的 SHA256 哈希值，并尝试获取 Git 提交信息，目前感觉没必要
    root = Path(__file__).resolve().parent.parent
    files = sorted(p for folder in ("kvlab", "configs", "tests", "scripts", "docs")
                   for p in (root / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    files += [p for p in (root / "pyproject.toml", root / "README.md") if p.exists()]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    sha = hashlib.sha256("".join(f"{k}:{v}\n" for k, v in sorted(hashes.items())).encode()).hexdigest()
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5).stdout.strip() or None
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        commit, dirty = None, None
    return {"git_commit": commit, "git_dirty": dirty, "source_sha256": sha, "file_sha256": hashes}
