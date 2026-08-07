"""Shared structural assertions for country-pack tests.

Each ``test_country_<cc>.py`` validates the structural invariants
shared by every concrete country jurisdiction overlay (US, DE, BR,
IN, JP, …). The file-level tests for a given country then layer the
country-specific severity-floor assertions on top of these shared
checks.
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

REQUIRED_TOP_LEVEL = {
    "skill_id",
    "parent",
    "schema_version",
    "expires_on",
    "signers",
    "activation",
    "local_definitions",
    "local_language_assets",
    "overrides",
    "allowed_contexts",
    "user_notice",
}

REQUIRED_FORBIDDEN_CRITERIA = {
    "gps_location",
    "ip_geolocation",
    "inferred_nationality",
    "inferred_ethnicity",
    "inferred_religion",
}

REQUIRED_SIGNERS = {"trust_and_safety", "legal_review", "cultural_review"}

REQUIRED_ALLOWED_CONTEXTS = {
    "QUOTED_SPEECH_CONTEXT",
    "NEWS_CONTEXT",
    "EDUCATION_CONTEXT",
    "COUNTERSPEECH_CONTEXT",
}


def as_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return datetime.strptime(value, "%Y-%m-%d").date()
    raise TypeError(f"unsupported expires_on type: {type(value).__name__}")


def get_override(overlay: dict, category: int) -> dict:
    matches = [o for o in overlay["overrides"] if o.get("category") == category]
    assert matches, f"overlay must define an override for category {category}"
    assert len(matches) == 1, (
        f"exactly one override expected for category {category}; got {len(matches)}"
    )
    return matches[0]


def assert_required_top_level(overlay: dict) -> None:
    missing = REQUIRED_TOP_LEVEL - set(overlay.keys())
    assert not missing, f"overlay missing top-level keys: {missing}"


def assert_skill_id(overlay: dict, country_code: str) -> None:
    skill_id = overlay["skill_id"]
    assert skill_id == (
        f"kchat.jurisdiction.{country_code}.guardrail.v1"
    ), f"unexpected skill_id: {skill_id!r}"
    assert skill_id.startswith("kchat.jurisdiction.")
    assert skill_id.endswith(".guardrail.v1")


def assert_parent_is_global_baseline(overlay: dict) -> None:
    assert overlay["parent"] == "kchat.global.guardrail.baseline"


def assert_schema_version_is_1(overlay: dict) -> None:
    assert overlay["schema_version"] == 1


def assert_signers(overlay: dict) -> None:
    signers = set(overlay["signers"])
    missing = REQUIRED_SIGNERS - signers
    assert not missing, f"overlay signers missing: {missing}"


def assert_forbidden_criteria(overlay: dict) -> None:
    forbidden = set(overlay["activation"]["forbidden_criteria"])
    missing = REQUIRED_FORBIDDEN_CRITERIA - forbidden
    assert not missing, f"overlay forbidden_criteria missing: {missing}"


def assert_activation_criteria(overlay: dict, country_code: str) -> None:
    criteria = overlay["activation"]["criteria"]
    assert isinstance(criteria, list) and len(criteria) >= 1
    # Every concrete country pack must drive activation off its own
    # region code.
    flat = [
        v
        for entry in criteria
        if isinstance(entry, dict)
        for v in entry.values()
    ]
    assert country_code in flat, (
        f"activation criteria do not reference country code {country_code!r}: "
        f"{criteria}"
    )


def assert_allowed_contexts(overlay: dict) -> None:
    contexts = set(overlay["allowed_contexts"])
    missing = REQUIRED_ALLOWED_CONTEXTS - contexts
    assert not missing, f"allowed_contexts missing: {missing}"


def assert_no_relaxed_child_safety(overlay: dict) -> None:
    for override in overlay["overrides"]:
        if override.get("category") == 1:
            assert override["severity_floor"] >= 5, (
                "CHILD_SAFETY floor cannot be lowered below 5"
            )


def assert_expiry_within_18_months(overlay: dict) -> None:
    expires = as_date(overlay["expires_on"])
    today = date.today()
    assert expires > today, "expires_on must be in the future"
    delta_days = (expires - today).days
    assert delta_days <= 18 * 31, (
        f"expires_on must be <= ~18 months from today; got {delta_days} days"
    )


def assert_user_notice(overlay: dict) -> None:
    notice = overlay["user_notice"]
    assert isinstance(notice.get("visible_pack_summary"), str)
    assert notice["visible_pack_summary"].strip()
    assert "appeal_resource_id" in notice
    assert "opt_out_allowed" in notice
    assert isinstance(notice["opt_out_allowed"], bool)


def assert_normalization(overlay: dict) -> None:
    norm = overlay["local_language_assets"]["normalization"]
    assert norm["nfkc"] is True
    assert norm["case_fold"] is True
    assert norm["homoglyph_map_id"]
    assert isinstance(norm["transliteration_refs"], list)
    assert norm["transliteration_refs"], (
        "transliteration_refs must be a non-empty list"
    )


def assert_normalization_file_matches(
    overlay: dict, normalization_file: dict
) -> None:
    overlay_norm = overlay["local_language_assets"]["normalization"]
    assert normalization_file["nfkc"] is True
    assert normalization_file["case_fold"] is True
    assert (
        normalization_file["homoglyph_map_id"]
        == overlay_norm["homoglyph_map_id"]
    )
    assert (
        list(normalization_file["transliteration_refs"])
        == list(overlay_norm["transliteration_refs"])
    )


def assert_lexicons_directory_exists(
    jurisdictions_dir: Path, country_code: str
) -> None:
    lexicons = jurisdictions_dir / country_code / "lexicons"
    assert lexicons.is_dir(), (
        f"jurisdiction '{country_code}' must ship a lexicons/ subdirectory"
    )
    yamls = list(lexicons.glob("*.yaml")) + list(lexicons.glob("*.yml"))
    assert yamls, (
        f"jurisdiction '{country_code}' lexicons/ must contain at least "
        "one YAML lexicon file"
    )


def assert_lexicons_have_provenance(overlay: dict) -> None:
    lexicons = overlay["local_language_assets"]["lexicons"] or []
    assert lexicons, "overlay must declare at least one lexicon"
    for lex in lexicons:
        assert lex.get("provenance"), (
            f"lexicon {lex.get('lexicon_id')!r} missing provenance"
        )
        assert lex.get("language"), (
            f"lexicon {lex.get('lexicon_id')!r} missing language"
        )
        assert lex.get("categories"), (
            f"lexicon {lex.get('lexicon_id')!r} missing categories"
        )


def run_all_structural_assertions(
    overlay: dict,
    *,
    country_code: str,
    normalization_file: dict,
    jurisdictions_dir: Path,
) -> None:
    """Convenience aggregator — runs every shared structural check."""
    assert_required_top_level(overlay)
    assert_skill_id(overlay, country_code)
    assert_parent_is_global_baseline(overlay)
    assert_schema_version_is_1(overlay)
    assert_signers(overlay)
    assert_forbidden_criteria(overlay)
    assert_activation_criteria(overlay, country_code)
    assert_allowed_contexts(overlay)
    assert_no_relaxed_child_safety(overlay)
    assert_expiry_within_18_months(overlay)
    assert_user_notice(overlay)
    assert_normalization(overlay)
    assert_normalization_file_matches(overlay, normalization_file)
    assert_lexicons_directory_exists(jurisdictions_dir, country_code)
    assert_lexicons_have_provenance(overlay)
