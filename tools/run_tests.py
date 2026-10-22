"""Minimal test runner used when pytest is unavailable.

Tests are written for pytest. This runner provides a tiny stand-in for the
subset of the pytest API they use (raises, mark.parametrize, approx, skip) so
that they can be executed in a bare Python environment:

    python tools/run_tests.py
"""

from __future__ import annotations

import importlib.util
import inspect
import math
import sys
import traceback
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class _Skip(Exception):
    pass


class _Raises:
    def __init__(self, exc, match=None):
        self.exc, self.match = exc, match

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            raise AssertionError(f"DID NOT RAISE {self.exc}")
        return issubclass(et, self.exc)


class _Approx:
    __array_ufunc__ = None  # make numpy defer comparisons to this object
    __array_priority__ = 100

    def __init__(self, expected, rel=1e-6, abs=1e-12):
        self.expected, self.rel, self.abs = expected, rel, abs

    def __eq__(self, other):
        exp = self.expected
        if isinstance(exp, (list, tuple)):
            return all(_Approx(e, self.rel, self.abs) == o for e, o in zip(exp, other))
        return math.isclose(float(other), float(exp), rel_tol=self.rel, abs_tol=self.abs)


def _parametrize(names, values):
    names = [n.strip() for n in names.split(",")] if isinstance(names, str) else list(names)

    def deco(fn):
        fn._params = (names, values)
        return fn

    return deco


def _importorskip(name):
    try:
        return __import__(name)
    except ImportError as e:
        raise _Skip(f"{name} not installed") from e


stub = types.ModuleType("pytest")
stub.raises = lambda exc, match=None: _Raises(exc, match)
stub.approx = _Approx
stub.mark = types.SimpleNamespace(parametrize=_parametrize, slow=lambda f: f)
stub.skip = lambda reason="": (_ for _ in ()).throw(_Skip(reason))
stub.importorskip = _importorskip
sys.modules.setdefault("pytest", stub)


def main() -> int:
    failures, passed, skipped = 0, 0, 0
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except _Skip as e:
            print(f"SKIP {path.name}: {e}")
            skipped += 1
            continue
        for name, fn in inspect.getmembers(mod, inspect.isfunction):
            if not name.startswith("test_") or fn.__module__ != mod.__name__:
                continue
            cases = [((), {})]
            if hasattr(fn, "_params"):
                names, values = fn._params
                cases = [((), dict(zip(names, v if len(names) > 1 else (v,)))) for v in values]
            for _, kwargs in cases:
                label = f"{path.name}::{name}" + (f"[{list(kwargs.values())[0]}]" if kwargs else "")
                try:
                    fn(**kwargs)
                    passed += 1
                except _Skip as e:
                    skipped += 1
                    print(f"SKIP {label}: {e}")
                except Exception:
                    failures += 1
                    print(f"FAIL {label}")
                    traceback.print_exc()
    print(f"\n{passed} passed, {failures} failed, {skipped} skipped")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
