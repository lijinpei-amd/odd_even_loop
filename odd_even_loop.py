from functools import cache
from pathlib import Path
import shutil
import subprocess

import torch
from torch.utils.cpp_extension import library_paths, load

THIS_DIR = Path(__file__).resolve().parent
SOURCE = THIS_DIR / "odd_even_loop.cpp"
BUILD_DIR = THIS_DIR / "torch_ext_build"
HSACO_BUILD_DIR = BUILD_DIR / "hsaco"

DEFAULT_ASM = THIS_DIR / "same_loop_even.s"
ORIGINAL_ASM = THIS_DIR / "original.s"

ORIGINAL_KERNEL_NAME = (
    "Custom_Cijk_Alik_Bljk_BBS_BH_MT256x256x64_MI16x16x1_UserArgs_shortname1_gfx950"
)
SAME_LOOP_KERNEL_NAME = "same_even_loop_" + ORIGINAL_KERNEL_NAME


@cache
def load_ext(verbose: bool = False):
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    torch_rpaths = [f"-Wl,-rpath,{path}" for path in library_paths()]
    return load(
        name="odd_even_loop",
        sources=[str(SOURCE)],
        build_directory=str(BUILD_DIR),
        extra_cflags=["-O3", "-std=c++17", "-D__HIP_PLATFORM_AMD__"],
        extra_ldflags=[
            "-L/opt/rocm/lib",
            "-Wl,-rpath,/opt/rocm/lib",
            *torch_rpaths,
            "-lamdhip64",
        ],
        with_cuda=True,
        verbose=verbose,
    )


def _clang() -> str:
    candidates = [
        "/opt/rocm-7.2.0/lib/llvm/bin/clang",
        "/opt/rocm/lib/llvm/bin/clang",
        "/opt/rocm/bin/clang",
        shutil.which("clang"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    raise RuntimeError("could not find ROCm clang")


def jit_hsaco(asm_path: str | Path, hsaco_path: str | Path | None = None) -> Path:
    asm_path = Path(asm_path)
    if hsaco_path is None:
        hsaco_path = HSACO_BUILD_DIR / f"{asm_path.stem}.hsaco"
    hsaco_path = Path(hsaco_path)
    obj_path = hsaco_path.with_suffix(".o")

    if hsaco_path.exists() and hsaco_path.stat().st_mtime >= asm_path.stat().st_mtime:
        return hsaco_path

    HSACO_BUILD_DIR.mkdir(parents=True, exist_ok=True)
    clang = _clang()
    common = [
        clang,
        "-target",
        "amdgcn-amd-amdhsa",
        "-mcpu=gfx950",
        "-mcode-object-version=5",
    ]
    subprocess.run(
        [*common, "-x", "assembler", "-c", str(asm_path), "-o", str(obj_path)],
        check=True,
    )
    subprocess.run(
        [*common, "-nostdlib", "-shared", str(obj_path), "-o", str(hsaco_path)],
        check=True,
    )
    return hsaco_path


@cache
def _resolved_hsaco(asm_path: Path) -> str:
    return str(jit_hsaco(asm_path))


def same_loop_even_hsaco() -> str:
    return _resolved_hsaco(DEFAULT_ASM)


def original_hsaco() -> str:
    return _resolved_hsaco(ORIGINAL_ASM)


def empty_output(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return load_ext().empty_output(a, b)


def matmul_same_loop(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return load_ext().matmul(a, b, same_loop_even_hsaco(), SAME_LOOP_KERNEL_NAME)


def matmul_same_loop_out(
    a: torch.Tensor, b: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    return load_ext().matmul_out(
        a, b, out, same_loop_even_hsaco(), SAME_LOOP_KERNEL_NAME
    )


def matmul_original(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return load_ext().matmul(a, b, original_hsaco(), ORIGINAL_KERNEL_NAME)


def matmul_original_out(
    a: torch.Tensor, b: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    return load_ext().matmul_out(a, b, out, original_hsaco(), ORIGINAL_KERNEL_NAME)
