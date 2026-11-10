"""Local editable installation of an already-built CRAFT-GBM checkout.

Binary distributions need their own platform/runtime verification. Refuse ordinary
wheel and sdist builds instead of accidentally distributing an incomplete package.
"""
from pathlib import Path

from setuptools import build_meta as _backend


def _check_build():
    root = Path(__file__).resolve().parent
    package = root / "python-package" / "lightgbm"
    if not (package / "lib" / "lib_lightgbm.so").is_file() or not (package / "VERSION.txt").is_file():
        raise RuntimeError(
            "Build CRAFT-GBM first: python scripts/build_native.py --jobs 4. "
            "Then run python -m pip install --no-deps -e ."
        )


def get_requires_for_build_editable(config_settings=None):
    _check_build()
    return _backend.get_requires_for_build_editable(config_settings)


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    _check_build()
    return _backend.prepare_metadata_for_build_editable(metadata_directory, config_settings)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    _check_build()
    return _backend.build_editable(wheel_directory, config_settings, metadata_directory)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    raise RuntimeError(
        "This checkout supports local editable installation only: "
        "build the native library, then python -m pip install --no-deps -e . "
        "Distributable binary wheels are not provided."
    )


def build_sdist(sdist_directory, config_settings=None):
    raise RuntimeError("Use the GitHub source checkout and the documented editable installation.")
