"""Anti-inverse-crime guard.

The code worlds (`worlds/`) and the shared core (`cawm/`) must be written without the simulators
that play the real world.  This test fails if any module there imports VizDoom, PyBullet, PyFlyt,
MuJoCo, Gymnasium or the real-world lab package -- directly or via importlib / __import__.

Only the bridge scripts under `experiments/` may import both a code world and `reallab`.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUARDED = ["worlds", "cawm"]
FORBIDDEN = {"vizdoom", "pybullet", "pybullet_data", "pybullet_utils", "PyFlyt", "mujoco", "gymnasium", "gym",
             "reallab"}


def _imports(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name, node.lineno
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module, node.lineno
        elif isinstance(node, ast.Call):
            f = node.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
            if name in ("__import__", "import_module") and node.args and isinstance(node.args[0], ast.Constant):
                yield str(node.args[0].value), node.lineno


def test_code_worlds_do_not_import_the_real_world():
    bad = []
    for top in GUARDED:
        for path in sorted((ROOT / top).rglob("*.py")):
            for mod, line in _imports(ast.parse(path.read_text(), filename=str(path))):
                if mod.split(".")[0] in FORBIDDEN:
                    bad.append(f"{path.relative_to(ROOT)}:{line} imports {mod}")
    assert not bad, "real-world simulator leaked into the code world:\n" + "\n".join(bad)


def test_code_worlds_do_not_read_lab_files():
    """Crude textual check: no code-world file may mention lab-internal paths or simulator asset files."""
    needles = ("reallab/", "reallab.", ".wad", ".urdf", "scenarios_path")
    bad = []
    for top in GUARDED:
        for path in sorted((ROOT / top).rglob("*.py")):
            if path.name == "ledger.py":
                continue
            text = path.read_text()
            for n in needles:
                if n in text:
                    bad.append(f"{path.relative_to(ROOT)} mentions {n!r}")
    assert not bad, "\n".join(bad)
