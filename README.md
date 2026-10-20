# DTRC-TOI

Dual-tolerance residual-compensated cumulative TOI termination, two-segment
CUDA Graph PCG, and indexed matrix-converter reduction.

双容差残差补偿 TOI＋Graph＋indexed（C0，残差阈值 0.03）。

## Source version

- Upstream: [Stiff-GIPC](https://github.com/KemengHuang/Stiff-GIPC), commit
  `c49982fac7767225c7775683734c89eb74081142`.
- Frozen shared implementation:
  [`3a45087fdbd3b0f1ca6e12ef2de3ab91b0db42df`](https://github.com/youyu210/DTRC-TOI/commit/3a45087fdbd3b0f1ca6e12ef2de3ab91b0db42df).
- Tag `implementation-3a45087f` points to that exact implementation. The
  publication commit on `main` only adds this README introduction; simulation
  sources, assets, and build files are unchanged from the frozen commit.

## Build and run C0

Install a compatible NVIDIA CUDA toolkit, a C++17 compiler, CMake, Eigen3,
GLEW, FreeGLUT/OpenGL, and nlohmann-json. Configure their package paths (or
your vcpkg toolchain), then build from this repository:

```sh
cmake -S . -B build
cmake --build build --config Release
```

Use the following explicit options for C0. The same executable also contains
baseline and ablation modes, so its default options do not select C0.

```sh
./build/gipc --scene 1 --benchmark-frames 10 --benchmark-output-dir Output/c0-smoke --termination compensated --toi-min-updates 6 --toi-threshold 0.001 --residual-threshold 0.03 --residual-floor 1e-30 --pcg-dual-graph --converter-indexed-reduce --newton-tolerance-scale 1 --pcg-tolerance-scale 1
```

On Windows, substitute `build\gipc.exe` or `build\Release\gipc.exe`, according
to the CMake generator. Use a new output directory for each run. The retained
`BUILD_WINDOWS.cmd` and `RUN_WINDOWS.*` scripts contain paths/configuration
from the original development machine; adapt those paths before using them.
Assets and output paths are embedded at build time, so rebuild after relocating
the source tree.

The two tolerances are the cumulative TOI tolerance (`0.001`) and normalized
nonlinear residual tolerance (`0.03`). The original direction-based stopping
rule remains active and takes priority. `toi` and `gated` are ablation modes;
`compensated` retains the residual-compensation history. Graph and indexed
reduction must both be enabled explicitly.

## Retained material

This upload preserves the frozen implementation's Git history, sources, input
assets, tests, and tracked experiment records. Later local diagnostics and
uncommitted experiments are outside this version. Executables, dependency DLLs,
and ignored binary experiment outputs are not included; build from source.
Historical READMEs and experiment reports describe their own versions and
settings; use the C0 command above for the requested configuration.

The existing `tests/toi_budget_test.cpp` checks compensation invariants and
dual-tolerance behavior. It passed with MSVC during publication on 2026-10-13.
The full CUDA application was not rebuilt or rerun as part of this upload.

Original attribution, license notice, and citation information follow.

---

# Unified GPU IPC Framework


DESCRIPTION
===========

This repository contains the source code for StiffGIPC, a Unified GPU Incremental Potential Contact Framework introduced in our paper: [StiffGIPC: Advancing GPU IPC for Stiff Affine-Deformable Simulation](https://dl.acm.org/doi/10.1145/3735126) **ACM Transactions on Graphics, 2025**. Our framework consistently achieves high performance in soft, stiff, and hybrid simulations (cloth, elastic solids, rigid bodies, and their hybrid couplings), even when handling high-resolution models, large deformations, and high-speed impacts.


Source code contributor: [Kemeng Huang](https://kemenghuang.github.io), Xinyu Lu

**Note: this software is released under the MPLv2.0 license. For commercial use, please email the authors for negotiation.**

## video 1
[![Watch the video](https://github.com/KemengHuang/Stiff-GIPC/blob/main/Assets/teaser.JPG)](https://www.youtube.com/watch?v=3TBoTX2vag4&list=LL)

## BibTex 

Please cite the following papers if it helps. 

```
@article{stiffgipc2025,
      author = {Huang, Kemeng and Lu, Xinyu and Lin, Huancheng and Komura, Taku and Li, Minchen},
      title = {StiffGIPC: Advancing GPU IPC for Stiff Affine-Deformable Simulation},
      year = {2025},
      publisher = {Association for Computing Machinery},
      volume = {44},
      number = {3},
      issn = {0730-0301},
      doi = {10.1145/3735126},
      journal = {ACM Trans. Graph.},
      month = may,
      articleno = {31},
      numpages = {20}
}
```

```
@article{gipc2024,
      author = {Huang, Kemeng and Chitalu, Floyd M. and Lin, Huancheng and Komura, Taku},
      title = {GIPC: Fast and Stable Gauss-Newton Optimization of IPC Barrier Energy},
      year = {2024},
      publisher = {Association for Computing Machinery},
      volume = {43},
      number = {2},
      issn = {0730-0301},
      doi = {10.1145/3643028},
      journal = {ACM Trans. Graph.},
      month = {mar},
      articleno = {23},
      numpages = {18}
}
```





Requirements
============

Hardware requirements: Nvidia GPUs

Support platforms: Windows, Linux 

## Dependencies

| Name                                   | Version | Usage                                               | Import         |
| -------------------------------------- | ------- | --------------------------------------------------- | -------------- |
| cuda                                   | >=11.0  | GPU programming                                     | system install |
| eigen3                                 | 3.4.0   | matrix calculation                                  | package        |
| freeglut                               | 3.4.0   | visualization                                       | package        |
| glew                                   | 2.2.0#3 | visualization                                       | package        |

### linux

We use CMake to build the project.

```bash
sudo apt install libglew-dev freeglut3-dev libeigen3-dev nlohmann-json3-dev
```


### Windows && linux
We use [vcpkg](https://github.com/microsoft/vcpkg) to manage the libraries we need and use CMake to build the project. The simplest way to let CMake detect vcpkg is to set the system environment variable `CMAKE_TOOLCHAIN_FILE` to `(YOUR_VCPKG_PARENT_FOLDER)/vcpkg/scripts/buildsystems/vcpkg.cmake`

```shell
vcpkg install eigen3 freeglut glew freeglut nlohmann-json
```


EXTERNAL CREDITS
================

This work utilizes the following external software library, which have been included here for convenience:
Copyrights are retained by the original authors.

- **METIS**: https://github.com/KemengHuang/METIS (fork of https://github.com/KarypisLab/METIS)  
- **GKlib**: https://github.com/KemengHuang/GKlib (fork of https://github.com/KarypisLab/GKlib)

