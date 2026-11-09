"""Run the analysis' small synthetic checks without a pytest dependency."""
import importlib.util
import inspect
from pathlib import Path
import sys
import unittest


def main():
    suite = unittest.TestSuite()
    for path in sorted(Path(__file__).resolve().parent.glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = module
        spec.loader.exec_module(module)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
        for name, function in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("test_") and function.__module__ == module.__name__:
                if inspect.signature(function).parameters:
                    raise ValueError(f"{path.name}:{name} requires unsupported fixtures")
                suite.addTest(unittest.FunctionTestCase(function, description=f"{path.name}:{name}"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
