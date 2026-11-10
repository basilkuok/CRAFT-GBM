"""Focused installation diagnostics; no native build or environment changes."""
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("craft_check_install", ROOT / "scripts/check_install.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class InstallationTests(unittest.TestCase):
    def setUp(self):
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        self.context.enter_context(patch.object(check.platform, "system", return_value="Linux"))
        self.context.enter_context(patch.object(check.sys, "version_info", (3, 12, 0, "final", 0)))
        self.context.enter_context(patch.object(check.sys, "prefix", str(ROOT / ".test-environment")))
        self.context.enter_context(patch.object(check.sys, "base_prefix", "/test/base-python"))

    def test_stock_lightgbm_distribution_reports_environment_conflict(self):
        stock = SimpleNamespace(locate_file=lambda name: "/other/site-packages")
        with patch.object(check.importlib.metadata, "distribution", return_value=stock):
            with self.assertRaisesRegex(check.InstallationError, "fresh virtual environment") as error:
                check.preflight(ROOT)
        self.assertIn("separate lightgbm distribution", str(error.exception))
        self.assertIn("/other/site-packages", str(error.exception))

    def test_foreign_python_import_reports_path_conflict(self):
        foreign = "/other/checkout/lightgbm/__init__.py"
        with patch.object(check.importlib.metadata, "distribution",
                          side_effect=check.importlib.metadata.PackageNotFoundError), \
                patch.object(check.importlib.util, "find_spec", return_value=SimpleNamespace(origin=foreign)):
            with self.assertRaisesRegex(check.InstallationError, "conflicting PYTHONPATH") as error:
                check.preflight(ROOT)
        self.assertIn(foreign, str(error.exception))

    def test_native_library_from_another_checkout_is_rejected(self):
        package = ModuleType("lightgbm")
        package.__file__ = str(ROOT / "python-package/lightgbm/__init__.py")
        package.__path__ = []
        craft = ModuleType("lightgbm.craft")
        craft.train_craft = lambda *args, **kwargs: None
        craft._check_craft_capabilities = lambda: None
        libpath = ModuleType("lightgbm.libpath")
        foreign = "/other/checkout/lib_lightgbm.so"
        libpath._LIB = SimpleNamespace(_name=foreign)
        distribution = SimpleNamespace(
            read_text=lambda name: json.dumps({"url": ROOT.as_uri(), "dir_info": {"editable": True}}),
            version="1.1.0",
        )
        with patch.object(check, "preflight", return_value={}), \
                patch.object(check.importlib.metadata, "distribution", return_value=distribution), \
                patch.object(check.importlib.metadata, "version", return_value="installed"), \
                patch.dict(sys.modules, {"lightgbm": package, "lightgbm.craft": craft,
                                         "lightgbm.libpath": libpath}):
            with self.assertRaisesRegex(check.InstallationError, "Unexpected package/library combination") as error:
                check.check_install(ROOT)
        self.assertIn(foreign, str(error.exception))
        self.assertIn("Rebuild and install this checkout", str(error.exception))


if __name__ == "__main__":
    unittest.main()
