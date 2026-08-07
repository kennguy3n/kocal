"""Parent-level conftest for ``kchat-skills/tests/``.

The build-tools restructure moved the Python ``compiler``
package from ``kchat-skills/compiler/`` to ``build-tools/compiler/``
while leaving the YAML-corpus test trees (``jurisdictions/``,
``adversarial/``, ``communities/``) under ``kchat-skills/tests/``
because the cases exercise pack data that *also* lives under
``kchat-skills/`` (e.g. the adversarial corpus YAML, the per-country
overlays). A small number of those tests still import the compiler
to run normalised text through the pipeline -- e.g.
``kchat-skills/tests/adversarial/test_adversarial_corpus.py`` calls
``from compiler.pipeline import normalize_text`` to exercise the
Step-1 normalisation against every obfuscation technique.

Without an explicit ``sys.path`` insert here those tests work *only*
because ``pyproject.toml`` configures
``pythonpath = ["build-tools", "kchat-skills"]`` and pytest picks
that up when invoked from the repo root. Any invocation path that
bypasses the pyproject.toml-managed pythonpath (e.g. running pytest
from inside a deeply nested working directory, or driving the test
via a tool that doesn't auto-discover the pyproject) would raise
``ModuleNotFoundError: No module named 'compiler'``. The defensive
``sys.path`` insert below mirrors the pattern used by
``kchat-skills/eval/eval_runner.py::_ensure_compiler_on_path`` and
makes the compiler-import dependency self-documenting at the test
tree root instead of action-at-a-distance from ``pyproject.toml``.

The ``kchat-skills/`` parent is added too so that any future test
under this tree that wants to import a sibling package -- e.g.
``from eval.X import Y`` or ``from compiler.pipeline import …`` --
resolves consistently with the runner pattern.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BUILD_TOOLS = _REPO_ROOT / "build-tools"
_KCHAT_SKILLS = _REPO_ROOT / "kchat-skills"

for _parent in (_BUILD_TOOLS, _KCHAT_SKILLS):
    _parent_str = str(_parent)
    if _parent_str not in sys.path:
        sys.path.insert(0, _parent_str)
