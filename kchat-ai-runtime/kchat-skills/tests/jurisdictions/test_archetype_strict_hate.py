"""Validate the kchat.jurisdiction.archetype-strict-hate pack.

The pack defines the strict hate / extremism archetype: severity
floor 4-5 on categories 4 and 6, with explicit protected contexts so
counter-speech, news, and education remain unblocked.
"""
from __future__ import annotations

from datetime import date, datetime

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


def _as_date(value) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return datetime.strptime(value, "%Y-%m-%d").date()
    raise TypeError(f"unsupported expires_on type: {type(value).__name__}")


def _get_override(overlay: dict, category: int) -> dict:
    matches = [o for o in overlay["overrides"] if o.get("category") == category]
    assert matches, f"overlay must define an override for category {category}"
    assert len(matches) == 1
    return matches[0]


# ---------------------------------------------------------------------------
# Structural
# ---------------------------------------------------------------------------
def test_parses_as_valid_yaml(strict_hate_overlay):
    assert isinstance(strict_hate_overlay, dict)


def test_required_top_level_keys(strict_hate_overlay):
    missing = REQUIRED_TOP_LEVEL - set(strict_hate_overlay.keys())
    assert not missing, f"overlay missing top-level keys: {missing}"


def test_skill_id_matches_pattern(strict_hate_overlay):
    skill_id = strict_hate_overlay["skill_id"]
    assert skill_id == (
        "kchat.jurisdiction.archetype-strict-hate.guardrail.v1"
    )
    assert skill_id.startswith("kchat.jurisdiction.")
    assert skill_id.endswith(".guardrail.v1")


def test_parent_is_global_baseline(strict_hate_overlay):
    assert strict_hate_overlay["parent"] == "kchat.global.guardrail.baseline"


def test_schema_version_is_1(strict_hate_overlay):
    assert strict_hate_overlay["schema_version"] == 1


# ---- Signers --------------------------------------------------------------
def test_signers_include_legal_and_cultural_review(strict_hate_overlay):
    signers = set(strict_hate_overlay["signers"])
    missing = REQUIRED_SIGNERS - signers
    assert not missing


# ---- Activation -----------------------------------------------------------
def test_forbidden_criteria_has_all_five(strict_hate_overlay):
    forbidden = set(strict_hate_overlay["activation"]["forbidden_criteria"])
    missing = REQUIRED_FORBIDDEN_CRITERIA - forbidden
    assert not missing


# ---- Overrides: category 4 floor in {4,5}, category 6 floor in {4,5} ------
def test_category_4_override_has_severity_floor_4_or_5(strict_hate_overlay):
    override = _get_override(strict_hate_overlay, 4)
    assert override["severity_floor"] in (4, 5), (
        "category 4 (EXTREMISM) override must be severity floor 4 or 5"
    )


def test_category_6_override_has_severity_floor_4_or_5(strict_hate_overlay):
    override = _get_override(strict_hate_overlay, 6)
    assert override["severity_floor"] in (4, 5), (
        "category 6 (HATE) override must be severity floor 4 or 5"
    )


def test_no_override_relaxes_child_safety_floor(strict_hate_overlay):
    for override in strict_hate_overlay["overrides"]:
        if override.get("category") == 1:
            assert override["severity_floor"] >= 5


# ---- Allowed contexts (explicit in ARCHITECTURE.md for this archetype) ---------
def test_allowed_contexts_include_all_four_protected_speech_contexts(
    strict_hate_overlay,
):
    contexts = set(strict_hate_overlay["allowed_contexts"])
    missing = REQUIRED_ALLOWED_CONTEXTS - contexts
    assert not missing, (
        "archetype-strict-hate must explicitly declare all 4 "
        f"protected-speech contexts; missing: {missing}"
    )


# ---- Expiry ---------------------------------------------------------------
def test_expires_on_is_within_18_months(strict_hate_overlay):
    expires = _as_date(strict_hate_overlay["expires_on"])
    today = date.today()
    assert expires > today
    assert (expires - today).days <= 18 * 31


# ---- User notice ----------------------------------------------------------
def test_user_notice_has_summary(strict_hate_overlay):
    notice = strict_hate_overlay["user_notice"]
    assert isinstance(notice.get("visible_pack_summary"), str)
    assert notice["visible_pack_summary"].strip()
    assert "appeal_resource_id" in notice
    assert "opt_out_allowed" in notice


# ---- Local language assets / normalization ------------------------------
def test_local_language_assets_have_normalization(strict_hate_overlay):
    norm = strict_hate_overlay["local_language_assets"]["normalization"]
    assert norm["nfkc"] is True
    assert norm["case_fold"] is True
    assert norm["homoglyph_map_id"]
    assert isinstance(norm["transliteration_refs"], list)


def test_normalization_file_matches_overlay(
    strict_hate_normalization, strict_hate_overlay
):
    overlay_norm = strict_hate_overlay["local_language_assets"]["normalization"]
    assert strict_hate_normalization["nfkc"] is True
    assert strict_hate_normalization["case_fold"] is True
    assert (
        strict_hate_normalization["homoglyph_map_id"]
        == overlay_norm["homoglyph_map_id"]
    )


# ---- Directory structure --------------------------------------------------
def test_lexicons_directory_exists(jurisdictions_dir):
    lexicons = jurisdictions_dir / "archetype-strict-hate" / "lexicons"
    assert lexicons.is_dir()
