"""AST-based architecture and code-quality checks (Requirement 27)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_PACKAGE = _ROOT / "mozzavid"
_FIJI_SCRIPT = _ROOT / "scripts" / "discover_fiji.py"


def _module_imports(path: Path) -> set[str]:
    """Return the fully-qualified module names imported by a Python file.

    `from pkg import name` is recorded as `pkg.name` so submodule imports are detected the
    same way whether written as `import pkg.name` or `from pkg import name`.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def _imports_target(path: Path, target: str) -> bool:
    """True if `path` imports `target` as a module or `from target import ...`."""
    return any(name == target or name.startswith(target + ".") for name in _module_imports(path))


def test_fiji_imported_only_by_allowed_modules():
    """mozzavid.fiji is imported only by preprocess, segment, measure and discover_fiji (Req 27.1)."""
    allowed = {"preprocess.py", "segment.py", "measure.py"}
    for path in _PACKAGE.glob("*.py"):
        if path.name in allowed or path.name == "fiji.py":
            continue
        assert not _imports_target(path, "mozzavid.fiji"), f"{path.name} imports mozzavid.fiji"
    assert _imports_target(_FIJI_SCRIPT, "mozzavid.fiji")


@pytest.mark.parametrize("module", ["imagej", "scyjava"])
def test_imagej_and_scyjava_only_in_fiji(module):
    """imagej and scyjava are imported only by fiji.py (Requirement 27.2)."""
    for path in _PACKAGE.glob("*.py"):
        if path.name == "fiji.py":
            continue
        assert not _imports_target(path, module), f"{path.name} imports {module}"


def test_xarray_only_in_fiji():
    """xarray is imported only by fiji.py (Requirement 24.4)."""
    for path in _PACKAGE.glob("*.py"):
        if path.name == "fiji.py":
            continue
        assert not _imports_target(path, "xarray"), f"{path.name} imports xarray"


def test_pipeline_does_not_import_synthetic():
    """pipeline.py does not import synthetic, avoiding an import cycle (Requirement 27.11)."""
    assert not _imports_target(_PACKAGE / "pipeline.py", "mozzavid.synthetic")


def test_no_skimage_or_hypothesis_anywhere():
    """Neither scikit-image nor hypothesis appears in any source or environment file (Req 27.10)."""
    for path in list(_PACKAGE.glob("*.py")) + list((_ROOT / "scripts").glob("*.py")):
        imports = _module_imports(path)
        assert "skimage" not in imports and "hypothesis" not in imports, path.name
    env_lines = [line.split("#", 1)[0].strip().lower()
                 for line in (_ROOT / "environment.yml").read_text(encoding="utf-8").splitlines()]
    deps = [line for line in env_lines if line.startswith("-")]
    assert not any("scikit-image" in line or "hypothesis" in line for line in deps)


def _functions(path: Path):
    """Yield every function definition node in a file."""
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def test_every_function_body_at_most_20_lines():
    """Every function body is at most 20 lines, excluding the docstring (Requirement 27.5)."""
    offenders = []
    for path in list(_PACKAGE.glob("*.py")) + list((_ROOT / "scripts").glob("*.py")):
        for node in _functions(path):
            has_doc = ast.get_docstring(node) is not None
            body = node.body[1:] if has_doc else node.body
            if body and body[-1].end_lineno - body[0].lineno + 1 > 20:
                offenders.append(f"{path.name}:{node.name}")
    assert not offenders, offenders


def test_every_public_function_has_docstring():
    """Every public function has a docstring (Requirement 27.6)."""
    offenders = []
    for path in list(_PACKAGE.glob("*.py")) + list((_ROOT / "scripts").glob("*.py")):
        for node in _functions(path):
            if not node.name.startswith("_") and ast.get_docstring(node) is None:
                offenders.append(f"{path.name}:{node.name}")
    assert not offenders, offenders
