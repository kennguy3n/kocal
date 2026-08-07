"""Held-out bias audit runner.

Wraps :class:`compiler.bias_audit.BiasAuditor` around the same
held-out YAML fixtures the calibration runner uses, then produces a
structured per-protected-class / per-language false-positive report
that gates the pack signing pipeline. ARCHITECTURE.md "Anti-Misuse
Controls" lists per-class / per-language false-positive monitoring
as a shipping requirement; this runner is the CI surface that
enforces it.

The runner stands up the same :class:`GuardrailPipeline`
configuration the calibration runner uses (mock encoder + eval-time
default lexicons), runs every held-out case through it, projects
each result into a :class:`BiasAuditCase` tagged with the
protected-class id derived from the case's tags / explicit
``protected_class`` override, and feeds the projection into
:meth:`BiasAuditor.run_audit`.

Tag → protected-class mapping lives in :data:`TAG_TO_PROTECTED_CLASS`
below. The mapping is documented and stable; eval cases that need to
override it (e.g. a case whose tags do not imply a protected class
but whose subject matter does) can add an explicit
``protected_class: <id>`` field to their case YAML. The runner reads
that field first and falls back to the tag mapping only when the
field is absent.

Usage::

    python kchat-skills/eval/bias_audit_runner.py \\
        --benign kchat-skills/eval/held_out_benign.yaml \\
        --adversarial kchat-skills/eval/held_out_adversarial.yaml \\
        --output /tmp/bias_audit.json

Exit codes:

* ``0`` — every protected class and language is within the bias
  thresholds carried by :class:`BiasAuditor`.
* ``1`` — at least one group exceeded the per-group FP ceiling or
  the disparity-vs-mean ceiling. The JSON output enumerates flagged
  classes / languages.
* ``2`` — runner error (bad YAML, missing file, pipeline boot
  failure, PyYAML not installed, audit data below
  ``--min-cases``, pipeline exceptions detected when
  ``--allow-pipeline-errors`` is not set, etc).

The ``--min-cases`` gate prevents *vacuous* gate passes: an audit
that ran against zero (or near-zero) benign cases would trivially
pass without producing meaningful evidence. The default is sized
for the held-out fixtures shipped in the repo (52 benign cases) and
is overridable for downstream consumers that build their own bias
corpora.

The ``--allow-pipeline-errors`` flag controls how the runner reacts
to cases that raise during input hydration or pipeline classify().
The historic behaviour was a silent SAFE-shaped substitution, which
masks **systematic per-group pipeline failures**: e.g. if every
disability-tagged case happens to trigger an encoder bug, those
cases all receive a SAFE verdict, the per-group false-positive rate
for the disability group computes to 0%, and the audit passes
vacuously despite the encoder being totally broken for that group.
The default (``--allow-pipeline-errors`` unset) is therefore
fail-fast: any exception in the pipeline path produces exit code 2
so CI flags the broken cohort instead of laundering it into a
pass. The opt-in path is for operators who deliberately want the
conservative SAFE-substitution semantics (e.g. partial-corpus
exploration) -- in that mode the report's
``pipeline_exception_counts`` field records per-protected-class
counts so the masked failure remains visible in the artefact.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _ensure_compiler_on_path() -> None:
    """Make the ``compiler`` and ``eval`` packages importable.

    Mirrors the helper in ``eval_runner.py`` — see that file for the
    full rationale. Two trees must be added: ``build-tools/`` for the
    ``compiler`` package and ``kchat-skills/`` for the sibling ``eval``
    package this runner imports from (``_yaml_loader``,
    ``eval_runner``). Without the ``kchat-skills/`` insert,
    ``from eval.X import Y`` raises ``ModuleNotFoundError`` when this
    file is invoked as a script (because Python's default
    ``sys.path[0]`` is the script's own directory, which is the
    package directory ``kchat-skills/eval/`` — not its parent).
    """
    repo_root = Path(__file__).resolve().parent.parent.parent
    for parent in (repo_root / "build-tools", repo_root / "kchat-skills"):
        parent_str = str(parent)
        if parent_str not in sys.path:
            sys.path.insert(0, parent_str)


# ---------------------------------------------------------------------------
# Tag → protected-class mapping. Treat this as the canonical source of
# truth for which held-out tags imply which protected class under U.S.
# / EU anti-discrimination law (race, religion, national origin /
# language minority, sex / sexual orientation / gender identity,
# disability, political opinion). The mapping is intentionally
# inclusive — a tag that even partially implicates a protected class
# is mapped to that class so the auditor can group it. Tags that do
# not map remain in the ``unspecified`` bucket.
#
# The taxonomy below mirrors the protected-class taxonomy used by
# :class:`compiler.bias_audit.BiasAuditCase` (``race``, ``religion``,
# ``national_origin``, ``sex``, ``sexual_orientation``,
# ``gender_identity``, ``disability``, ``political_opinion``).
# ---------------------------------------------------------------------------
TAG_TO_PROTECTED_CLASS: dict[str, str] = {
    # Reclaimed-slur reclamation by minority communities (e.g. the
    # n-word, queer reclamation, disability reclamation). Most often
    # implicates race or sexual orientation; we route to "race" as
    # the modal case and rely on the explicit ``protected_class``
    # override for sexual-orientation-specific cases.
    "reclaimed_language": "race",
    # Identity-disclosure / coming-out content. The encoder has
    # historically false-fired on this content as self-harm /
    # protected-speech. Route to sexual_orientation as the modal
    # protected class for coming-out tagging.
    "identity_discussion": "sexual_orientation",
    # Counter-speech against hate / bullying. The protected class
    # depends on what the counter-speech responds to; we route to
    # race as the modal case (most counter-speech corpora cover
    # racial / ethnic hate). Cases that target a different protected
    # class can override.
    "counterspeech": "race",
    # Mental-health discussion, therapy progress, recovery stories.
    # The encoder has historically false-fired on this content as
    # self-harm. Protected under disability anti-discrimination law.
    "health_discussion": "disability",
    "self_health_discussion": "disability",
    # Political discourse / opinion. Protected under political-
    # opinion anti-discrimination law in EU jurisdictions and under
    # First Amendment law in the US.
    "political_discourse": "political_opinion",
    # News-coverage of harm events. Protected as press freedom.
    "news_coverage": "political_opinion",
    # Educational discussion of harm. Protected as academic freedom.
    "education": "political_opinion",
    # Minority-language coverage. Protected under national-origin /
    # language-minority anti-discrimination law.
    "minority_language": "national_origin",
}


def _derive_protected_class(case: dict[str, Any]) -> str:
    """Resolve a case's protected-class id.

    Resolution order:

    1. Explicit ``protected_class`` field on the case YAML wins.
    2. Otherwise, the first tag that maps to a protected class via
       :data:`TAG_TO_PROTECTED_CLASS` wins. Tags are scanned in
       insertion order so authors can hint at the intended class by
       listing the most-specific tag first.
    3. Otherwise ``"unspecified"``.
    """
    explicit = case.get("protected_class")
    if isinstance(explicit, str) and explicit:
        return explicit
    for tag in case.get("tags") or ():
        if tag in TAG_TO_PROTECTED_CLASS:
            return TAG_TO_PROTECTED_CLASS[tag]
    return "unspecified"


def _load_yaml(path: Path) -> dict[str, Any]:
    """Parse ``path`` as YAML and return the top-level mapping.

    Thin wrapper around :func:`eval._yaml_loader.load_yaml_mapping`,
    kept for back-compat with tests and external callers that import
    ``_load_yaml`` directly. The actual schema-validation logic lives
    in the shared module so both runners cannot drift.
    """
    _ensure_compiler_on_path()
    from eval._yaml_loader import load_yaml_mapping

    return load_yaml_mapping(path)


def _extract_cases(loaded: dict[str, Any], source: Path) -> list[dict[str, Any]]:
    """Pull the ``cases`` list out of a loaded YAML mapping and validate it.

    Thin wrapper around :func:`eval._yaml_loader.extract_cases`, kept
    for back-compat with tests and external callers that import
    ``_extract_cases`` directly. See the shared module's docstring for
    the full schema-validation rationale -- including ``tags``
    list-shape validation that closes the ``for tag in case.get(
    "tags") or ():`` :class:`TypeError` gap inside
    :func:`_derive_protected_class`'s defensive handler.
    """
    _ensure_compiler_on_path()
    from eval._yaml_loader import extract_cases

    return extract_cases(loaded, source)


def _run_pipeline_on_cases(
    pipeline: Any, cases: list[dict[str, Any]]
) -> tuple[
    list[tuple[dict[str, Any], dict[str, Any]]],
    dict[str, int],
]:
    """Run the pipeline on each case and return runs + per-group exception counts.

    Returns a ``(runs, exception_counts)`` pair:

    * ``runs`` -- the list of ``(case, verdict)`` tuples consumed by
      :func:`_project_to_bias_audit_cases`. Cases that raise during
      input hydration or ``pipeline.classify()`` receive a
      SAFE-shaped fallback verdict so they continue downstream with
      a clean ``predicted_category`` (the bias auditor cannot
      reason about category-99 sentinels).
    * ``exception_counts`` -- a ``{protected_class: count}`` map
      that surfaces the per-cohort failure rate to the caller.
      The SAFE-shaped fallback is *also* a bias signal: if every
      case in a protected class raises, that class's per-group FP
      rate computes to 0% vacuously and the audit would pass while
      masking a systematic failure. The caller uses this map to
      decide whether to refuse the pass.

    The try/except covers **both** the input-hydration step
    (``_build_pipeline_input``) and the ``pipeline.classify()``
    call, since either can raise for a malformed case YAML or a
    pipeline boot failure -- and both failure modes deserve the
    same per-group accounting.
    """
    # Local import so the module is importable even in environments
    # without the compiler package on sys.path (e.g. when the
    # runner is being introspected for CLI flags only). The helper
    # lives in ``eval_runner.py``; we reuse it directly to avoid
    # drifting two parallel input-hydration paths.
    _ensure_compiler_on_path()
    from eval.eval_runner import _build_pipeline_input

    out: list[tuple[dict[str, Any], dict[str, Any]]] = []
    exception_counts: dict[str, int] = {}
    for case in cases:
        try:
            pipeline_input = _build_pipeline_input(case)
            verdict = pipeline.classify(
                pipeline_input["message"], pipeline_input["context"]
            )
        except Exception:
            # SAFE-shaped fallback. We deliberately do NOT treat the
            # exception as a FP -- the bias auditor reads
            # ``predicted_category`` and we want a clean SAFE so the
            # case doesn't poison the per-group FP rate with a
            # category-99 record that no downstream metric understands.
            # The exception *is* however recorded in
            # ``exception_counts`` so that the runner can refuse to
            # emit a passing verdict when a cohort fails
            # systematically -- without this, an encoder bug that
            # only fires on disability-tagged cases would silently
            # compute as a 0% per-group FP rate.
            verdict = {
                "category": 0,
                "severity": 0,
                "confidence": 0.0,
                "reason_codes": [],
            }
            cohort = _derive_protected_class(case)
            exception_counts[cohort] = exception_counts.get(cohort, 0) + 1
        out.append((case, verdict))
    return out, exception_counts


def _project_to_bias_audit_cases(
    runs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> list[Any]:
    """Convert pipeline runs into :class:`BiasAuditCase` rows.

    Only cases whose YAML expectation declares an explicit
    ``expected.category`` are included — that field is what tells
    the bias auditor whether the ground truth is SAFE (0) or some
    harm class (non-zero). Adversarial cases in the current held-out
    suite use ``severity_at_least`` / ``reason_codes_must_include``
    instead of pinning a category and therefore contribute to recall
    metrics in the calibration runner, not to false-positive rates
    here. Including them with a defaulted SAFE category would
    incorrectly inflate the per-group FP rate; including them with a
    sentinel non-SAFE category would still distort per-group sample
    counts. Skipping them is the only semantically correct option
    given that this auditor measures FP rate exclusively.
    """
    _ensure_compiler_on_path()
    from compiler.bias_audit import BiasAuditCase

    out: list[Any] = []
    for case, verdict in runs:
        expected = case.get("expected") or {}
        if "category" not in expected:
            continue
        expected_category = expected["category"]
        if not isinstance(expected_category, int):
            # Defensive: a non-int ``category`` field is malformed
            # eval YAML, not a bias-audit data point.
            continue
        out.append(
            BiasAuditCase(
                case_id=str(case.get("case_id", "")),
                protected_class=_derive_protected_class(case),
                language=str(case.get("language", "und")),
                expected_category=expected_category,
                predicted_category=int(verdict.get("category", 0)),
                tags=tuple(case.get("tags") or ()),
            )
        )
    return out


def _build_pipeline(encoder_kind: str) -> Any:
    """Wire up the runtime ``GuardrailPipeline`` for the audit run.

    Encoder selection follows the calibration runner: ``stub``
    (default) uses :class:`MockEncoderAdapter`; ``xlmr`` boots
    :class:`XLMRAdapter` and requires the ONNX model on disk.

    The actual wiring lives in
    :func:`eval.eval_runner._build_pipeline`; this thin wrapper
    exists so the bias audit runner's tests can monkeypatch
    ``eval.bias_audit_runner._build_pipeline`` directly (mirrors
    every other call-site convention in this module) without having
    to reach into ``eval_runner``'s module namespace. The local
    import is aliased to ``_eval_build_pipeline`` to make it
    unambiguous which function is being called -- a same-name local
    import would shadow this wrapper's own binding and read like a
    recursive call to a future maintainer skimming the file.
    """
    _ensure_compiler_on_path()
    from eval.eval_runner import _build_pipeline as _eval_build_pipeline

    return _eval_build_pipeline(encoder_kind)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benign",
        type=Path,
        required=True,
        help="Path to the held-out benign YAML.",
    )
    parser.add_argument(
        "--adversarial",
        type=Path,
        required=True,
        help="Path to the held-out adversarial YAML.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the JSON bias-audit report.",
    )
    parser.add_argument(
        "--encoder",
        choices=("stub", "xlmr"),
        default="stub",
        help="Encoder adapter to use (default: stub).",
    )
    parser.add_argument(
        "--pack-id",
        type=str,
        default="kchat.guardrail.held_out",
        help=(
            "Identifier recorded in the report's ``pack_id`` field "
            "(default: kchat.guardrail.held_out)."
        ),
    )
    parser.add_argument(
        "--min-cases",
        type=int,
        default=10,
        help=(
            "Minimum number of audit cases the runner must observe "
            "before it will emit a passing verdict. Exists to prevent "
            "vacuous passes when the eval YAML has been accidentally "
            "emptied or when every case lacks an explicit "
            "``expected.category`` field. Defaults to 10 -- the "
            "shipped held-out fixtures carry 52 benign cases, so 10 "
            "is well below floor but well above zero. Use a smaller "
            "value only for one-off experiments; downstream packs "
            "should pin a per-pack floor."
        ),
    )
    parser.add_argument(
        "--allow-pipeline-errors",
        action="store_true",
        default=False,
        help=(
            "When set, the runner tolerates per-case exceptions in "
            "``_build_pipeline_input`` / ``pipeline.classify`` and "
            "substitutes a SAFE verdict instead of failing. The "
            "default (flag unset) is fail-fast: any pipeline "
            "exception produces exit code 2. Fail-fast is the "
            "correct default for a bias gate -- a systematic "
            "per-cohort failure (e.g. the encoder raising on every "
            "disability-tagged case) computes to a 0%% per-group "
            "false-positive rate under SAFE substitution and would "
            "silently pass the audit. Opt in only for partial-corpus "
            "exploration; the report's ``pipeline_exception_counts`` "
            "field still records the per-cohort counts in that mode "
            "so the masked failure remains visible in the artefact."
        ),
    )
    args = parser.parse_args(argv)

    try:
        pipeline = _build_pipeline(args.encoder)
    except Exception as exc:
        print(f"failed to build pipeline: {exc}", file=sys.stderr)
        return 2

    # The top-level reference to ``yaml.YAMLError`` requires the
    # module to be importable before the ``except`` clause is
    # compiled. ``_load_yaml`` performs its own import-and-handle
    # dance, but if we put a bare ``import yaml`` here we defeat
    # that pattern: PyYAML missing on the box would raise an
    # unhandled :class:`ImportError` instead of the documented
    # runner-error exit code. Wrap the import too so the contract
    # holds even on a stripped install.
    try:
        import yaml
    except ImportError as exc:
        print(f"PyYAML is required for the bias audit runner: {exc}", file=sys.stderr)
        return 2

    # ``yaml.YAMLError`` inherits directly from :class:`Exception`,
    # not from :class:`ValueError` -- the same pitfall
    # ``eval_runner._load_default_lexicons`` documents -- so malformed
    # YAML must be caught explicitly or the runner crashes with an
    # uncaught exception instead of producing the documented exit
    # code 2.
    try:
        benign = _extract_cases(_load_yaml(args.benign), args.benign)
        adversarial = _extract_cases(
            _load_yaml(args.adversarial), args.adversarial
        )
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"failed to load eval YAML: {exc}", file=sys.stderr)
        return 2

    cases = list(benign) + list(adversarial)
    runs, exception_counts = _run_pipeline_on_cases(pipeline, cases)
    audit_cases = _project_to_bias_audit_cases(runs)

    # Fail-fast on per-cohort pipeline failures unless explicitly
    # opted out. See the ``--allow-pipeline-errors`` help text and
    # the module docstring for the bias-signal rationale: SAFE
    # substitution on a systematically-failing cohort produces a
    # 0% per-group false-positive rate and silently passes the
    # audit. The exception counts are surfaced in the report
    # artefact regardless so even the opt-in soft-fail mode keeps
    # the masked failure visible.
    total_exceptions = sum(exception_counts.values())
    if total_exceptions and not args.allow_pipeline_errors:
        cohort_summary = ", ".join(
            f"{cohort}={count}"
            for cohort, count in sorted(exception_counts.items())
        )
        print(
            f"bias audit encountered {total_exceptions} pipeline "
            f"exception(s) across cohorts ({cohort_summary}); "
            "refusing to emit a verdict. A systematic per-cohort "
            "failure can mask bias by computing as 0% FP under "
            "SAFE substitution. Inspect the failing pipeline or "
            "re-run with --allow-pipeline-errors to record the "
            "counts in the artefact and continue.",
            file=sys.stderr,
        )
        return 2

    # Minimum-sample-size gate. Without this an empty
    # ``audit_cases`` (the YAML had no ``expected.category`` fields,
    # or every case was filtered out by the malformed-category
    # defence) would produce a trivially-passing report -- empty
    # per-class and per-language buckets evaluate to ``passed=True``
    # vacuously. That is exactly the failure mode that turns CI
    # green on a broken corpus, so we refuse to emit a passing
    # verdict when the audit had less evidence than the operator
    # asked for.
    if len(audit_cases) < args.min_cases:
        print(
            f"bias audit had {len(audit_cases)} cases, below the "
            f"--min-cases floor of {args.min_cases}; refusing to "
            "emit a verdict. Inspect the held-out YAML for missing "
            "``expected.category`` fields.",
            file=sys.stderr,
        )
        return 2

    _ensure_compiler_on_path()
    from compiler.bias_audit import BiasAuditor

    auditor = BiasAuditor()
    report = auditor.run_audit(audit_cases, pack_id=args.pack_id)

    # ``report.as_dict()`` already carries the canonical per-class /
    # per-language outcome (``passed``, ``flagged_classes``,
    # ``flagged_languages``, ``per_class_results``, etc.) at the top
    # level -- duplicating those fields inside a ``gates`` sub-dict
    # would invite drift between the two copies. Instead, ``gates``
    # records ONLY the threshold values that don't appear in the
    # canonical report so CI consumers (the bias-audit job's artefact
    # uploader, downstream dashboards) can pin against the exact
    # ceiling that produced the verdict without re-deriving it from
    # :class:`BiasAuditor` source.
    payload = report.as_dict()
    payload["n_cases"] = len(audit_cases)
    # Sort the exception-counts dict deterministically so the JSON
    # artefact remains diffable across runs.
    payload["pipeline_exception_counts"] = dict(
        sorted(exception_counts.items())
    )
    payload["gates"] = {
        "max_per_group_fp_rate": auditor.max_per_group_fp_rate,
        "max_disparity": auditor.max_disparity,
        "min_cases": args.min_cases,
        "allow_pipeline_errors": args.allow_pipeline_errors,
    }
    # record the
    # encoder identity in the report so downstream consumers can
    # distinguish stub-encoder reports (CI-side fairness-of-detector
    # gate) from XLM-R reports (encoder-pipeline encoder-drift gate).
    # The two are NOT interchangeable as baselines for each other:
    # comparing an XLM-R candidate against a stub baseline (or vice
    # versa) is apples-to-oranges and the resulting "regression"
    # signal carries no fairness meaning. ``tools/compare_bias_audits``
    # uses this field to refuse mismatched comparisons, and
    # ``tools/select_bias_audit_gate_mode`` uses it to auto-promote a
    # first-run XLM-R candidate as the proposed XLM-R baseline.
    payload["encoder_kind"] = args.encoder

    if args.output is not None:
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    else:
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")

    return 0 if report.passed else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
