#!/usr/bin/env python3
"""Build the released CPU library and make its bundled Python bindings importable."""
from pathlib import Path
import argparse
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--boost-root", type=Path)
    parser.add_argument("--with-cpp-tests", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    root = Path(__file__).resolve().parents[1]
    build = root / "build"
    command = ["cmake", "-S", str(root), "-B", str(build),
               "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_C_FLAGS_RELEASE=-O3 -DNDEBUG",
               "-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG", "-DUSE_OPENMP=ON",
               "-DUSE_GPU=OFF", "-DUSE_CUDA=OFF", "-DUSE_MPI=OFF",
               "-DUSE_TIMETAG=OFF", "-DBUILD_STATIC_LIB=OFF", "-DBUILD_CLI=ON",
               "-DBUILD_CPP_TEST=" + ("ON" if args.with_cpp_tests else "OFF")]
    if args.boost_root is not None:
        candidates = [args.boost_root.resolve() / "include", args.boost_root.resolve()]
        include = next((p for p in candidates if (p / "boost/version.hpp").is_file()), None)
        if include is None:
            parser.error("--boost-root must contain include/boost or boost headers")
        command.append("-DBoost_INCLUDE_DIR=" + str(include))
    subprocess.run(command, check=True)
    targets = ["lightgbm", "_lightgbm"]
    if args.with_cpp_tests:
        targets.append("testlightgbm")
    subprocess.run(["cmake", "--build", str(build), "--parallel", str(args.jobs),
                    "--target", *targets], check=True)
    library = next((root / name for name in ["lib_lightgbm.so", "lib_lightgbm.dylib", "lib_lightgbm.dll"]
                    if (root / name).is_file()), None)
    if library is None:
        raise RuntimeError("Build completed without the native shared library")
    package = root / "python-package/lightgbm"
    (package / "lib").mkdir(exist_ok=True)
    shutil.copy2(library, package / "lib" / library.name)
    shutil.copy2(root / "VERSION.txt", package / "VERSION.txt")
    print("Built CRAFT-GBM. Add python-package to PYTHONPATH before importing lightgbm.")


if __name__ == "__main__":
    main()
