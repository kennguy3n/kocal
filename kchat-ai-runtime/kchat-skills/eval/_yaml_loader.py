"""Shared YAML-loading + ``cases:`` schema validation for eval runners.

The eval runner (``eval_runner.py``) and the bias audit runner
(``bias_audit_runner.py``) both consume the same held-out YAML
fixtures. Before this module existed they shipped two
behaviourally-identical copies of ``_load_yaml`` and ``_extract_cases``;
a ``TestExtractCasesSymmetry`` parametrised test class pinned the
equivalence at the unit level, but the duplication was a maintenance
smell -- a future refactor that updated one copy could silently drift
the other and the symmetry pin was the only enforcement.

Consolidating into a single module gives:

* **One source of truth** -- both runners import the same callables, so
  schema validation cannot drift between the two consumers of the
  same YAML.
* **No circular-import risk** -- this module imports only the standard
  library (and ``yaml`` lazily); neither runner is referenced from
  here, so it sits cleanly below both in the dependency graph.
* **A single place to widen the schema** -- the next field that needs
  per-case validation (e.g. ``expected.category`` shape, ``tags``
  list-shape, ``language`` BCP-47 well-formedness) lands here once
  rather than being mirrored in two files.

The functions raise :class:`ValueError` for every malformed-shape
input. Both runners route this through their existing ``try / except
(OSError, ValueError, yaml.YAMLError)`` triple in ``main()``, which
produces the documented exit-code-2 contract.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    """Parse ``path`` as YAML and return the top-level mapping.

    ``yaml.safe_load`` is intentionally permissive -- it returns
    whatever Python value the document decodes to (``None`` for an
    empty file, a ``list`` for a top-level sequence, an ``int`` /
    ``str`` for a scalar). Every caller in the eval / bias-audit
    runners immediately treats the return value as a ``dict``
    (``.get("cases", ...)``, etc), which raises an uncaught
    :class:`AttributeError` on the non-mapping shapes and bypasses the
    documented exit-code-2 contract. Reject any non-mapping top-level
    value with a :class:`ValueError` that ``main()`` already catches.

    An empty file (``yaml.safe_load`` -> ``None``) returns an empty
    mapping for back-compat -- a YAML file with only comments or
    whitespace is a legitimate (if useless) input that should not
    crash the runner.
    """
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "PyYAML is required to run the eval / bias-audit runners."
        ) from exc
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"expected top-level YAML mapping in {path}, got "
            f"{type(raw).__name__}"
        )
    return raw


def extract_cases(loaded: dict[str, Any], source: Path) -> list[dict[str, Any]]:
    """Pull ``cases`` out of a loaded YAML mapping and validate the schema.

    Validates the full shape of every field that the downstream
    consumers in *both* runners read with iterator- or mapping-shaped
    operations. Each per-field validation is a real bug-class boundary:

    * **``cases:`` container** must be a list (or absent / null).
      Downstream code iterates with ``for case in cases:`` and would
      otherwise crash with an opaque ``TypeError`` inside the loop.
    * **Each ``cases[i]`` element** must be a mapping. Every consumer
      calls ``case.get(...)``; a string / scalar / sequence in this
      position raises :class:`AttributeError` deep inside the
      consumer, including *inside* the runner's defensive
      ``except Exception`` boundary -- which defeats the boundary's
      whole purpose.
    * **Each ``cases[i].tags`` field**, when present, must be a list /
      tuple of strings (or ``None``). The bias audit runner's
      ``_derive_protected_class`` iterates with ``for tag in case.get(
      "tags") or ():``, which raises :class:`TypeError` for a scalar
      ``tags: 42`` and :class:`AttributeError` for a mapping
      ``tags: {a: 1}``. Catching this at the boundary closes the
      "exception handler can itself raise" gap that the per-case
      defensive handler cannot.

    Validating at the YAML-loading boundary is the architecturally
    correct place: every later consumer can assume its contract is
    met instead of replicating the isinstance check in N call sites.
    Error messages name the offending case index and source path so
    an operator can locate the bad row without diffing the YAML.

    Raises :class:`ValueError` on every malformed-shape input; both
    runners' ``main()`` route this through ``try / except (OSError,
    ValueError, yaml.YAMLError)`` to exit code 2.
    """
    cases = loaded.get("cases")
    if cases is None:
        return []
    if not isinstance(cases, list):
        raise ValueError(
            f"expected ``cases`` field to be a list in {source}, got "
            f"{type(cases).__name__}"
        )
    out: list[dict[str, Any]] = []
    for idx, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(
                f"expected each entry of ``cases`` to be a YAML "
                f"mapping in {source}, got {type(case).__name__} at "
                f"index {idx}"
            )
        tags = case.get("tags")
        if tags is not None and not isinstance(tags, (list, tuple)):
            # ``str`` is iterable but a scalar string in ``tags`` is
            # almost certainly a typo (``tags: news_coverage`` instead
            # of ``tags: [news_coverage]``) and iterating it character
            # by character would produce phantom single-letter tags
            # that never map to a protected class -- a silent bug
            # worse than the noisy crash. Reject strings at the
            # boundary too.
            raise ValueError(
                f"expected ``cases[{idx}].tags`` to be a list / tuple "
                f"or absent in {source}, got {type(tags).__name__}"
            )
        out.append(case)
    return out
