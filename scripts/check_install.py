#!/usr/bin/env python3
"""Check the isolated Python environment and the CRAFT-GBM library it loads."""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import sys


class InstallationError(RuntimeError):
    """An installation problem with a user-actionable explanation."""


def preflight(root: Path) -> dict:
    """Check the interpreter and import conflicts without loading native code."""
    if platform.system() != "Linux" or sys.version_info[:2] != (3, 12):
        raise InstallationError("Use the supported Linux / Python 3.12 environment described in README.md.")
    if sys.prefix == sys.base_prefix:
        raise InstallationError(
            "Use an isolated environment: python3.12 -m venv .venv; "
            "then activate it with . .venv/bin/activate."
        )
    try:
        stock = importlib.metadata.distribution("lightgbm")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        raise InstallationError(
            f"A separate lightgbm distribution is installed at {stock.locate_file('')}. "
            "Create a fresh virtual environment for CRAFT-GBM; both use the lightgbm import name."
        )
    spec = importlib.util.find_spec("lightgbm")
    expected = root / "python-package" / "lightgbm" / "__init__.py"
    if spec is not None and (spec.origin is None or Path(spec.origin).resolve() != expected.resolve()):
        raise InstallationError(
            f"Python would import lightgbm from {spec.origin}, not {expected}. "
            "Use the CRAFT-GBM virtual environment and remove conflicting PYTHONPATH entries."
        )
    return {"python": sys.version.split()[0], "executable": sys.executable,
            "environment": sys.prefix, "checkout": str(root)}


def check_install(root: Path) -> dict:
    report = preflight(root)
    try:
        distribution = importlib.metadata.distribution("craft-gbm")
    except importlib.metadata.PackageNotFoundError as exc:
        raise InstallationError(
            "CRAFT-GBM is not registered in this environment. Build the native library, "
            "then run python -m pip install --no-deps -e . from the checkout."
        ) from exc
    direct_url = distribution.read_text("direct_url.json")
    installation = json.loads(direct_url) if direct_url else {}
    if (installation.get("url") != root.as_uri()
            or not installation.get("dir_info", {}).get("editable", False)):
        raise InstallationError(
            "The craft-gbm distribution points to a different checkout or is not editable. "
            "Run python -m pip install --no-deps -e . from this checkout in a fresh environment."
        )
    for requirement in ("numpy", "scipy", "narwhals"):
        try:
            importlib.metadata.version(requirement)
        except importlib.metadata.PackageNotFoundError as exc:
            raise InstallationError("Install the pinned dependencies: python -m pip install -r requirements.txt") from exc
    try:
        import lightgbm as lgb
        from lightgbm.craft import _check_craft_capabilities
        from lightgbm.libpath import _LIB
    except Exception as exc:
        # libpath raises a plain Exception when the library has not been built.
        raise InstallationError(
            f"Cannot load CRAFT-GBM: {exc}. Build with python scripts/build_native.py --jobs 4, "
            "install requirements.txt, and check the C++/OpenMP runtime troubleshooting in README.md."
        ) from exc
    package = Path(lgb.__file__).resolve()
    library = Path(_LIB._name).resolve()
    expected_package = root / "python-package" / "lightgbm"
    if package != expected_package / "__init__.py" or library != expected_package / "lib" / "lib_lightgbm.so":
        raise InstallationError(
            f"Unexpected package/library combination: {package}, {library}. "
            "Rebuild and install this checkout in a fresh environment."
        )
    try:
        _check_craft_capabilities()
    except RuntimeError as exc:
        raise InstallationError(str(exc)) from exc
    report.update({"craft_version": distribution.version, "lightgbm_base_version": lgb.__version__,
                   "python_package": str(package), "native_library": str(library),
                   "craft_capabilities": "available"})
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="check the environment before building/installing")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        report = preflight(root) if args.preflight else check_install(root)
    except InstallationError as exc:
        print(f"CRAFT-GBM installation check failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
