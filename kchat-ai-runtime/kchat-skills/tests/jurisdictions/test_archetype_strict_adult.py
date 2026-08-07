"""Validate the kchat.jurisdiction.archetype-strict-adult pack.

The pack defines the strict adult-content archetype: severity floor
5 on category 10 across jurisdictions that prohibit adult content
outright.
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


# ---------------------------------------------------------------------------
# Structural
# ---------------------------------------------------------------------------
def test_parses_as_valid_yaml(strict_adult_overlay):
    assert isinstance(strict_adult_overlay, dict)


def test_required_top_level_keys(strict_adult_overlay):
    missing = REQUIRED_TOP_LEVEL - set(strict_adult_overlay.keys())
    assert not missing, f"overlay missing top-level keys: {missing}"


def test_skill_id_matches_pattern(strict_adult_overlay):
    skill_id = strict_adult_overlay["skill_id"]
    assert skill_id == (
        "kchat.jurisdiction.archetype-strict-adult.guardrail.v1"
    )
    assert skill_id.startswith("kchat.jurisdiction.")
    assert skill_id.endswith(".guardrail.v1")


def test_parent_is_global_baseline(strict_adult_overlay):
    assert strict_adult_overlay["parent"] == "kchat.global.guardrail.baseline"


def test_schema_version_is_1(strict_adult_overlay):
    assert strict_adult_overlay["schema_version"] == 1


# ---- Signers --------------------------------------------------------------
def test_signers_include_legal_and_cultural_review(strict_adult_overlay):
    signers = set(strict_adult_overlay["signers"])
    missing = REQUIRED_SIGNERS - signers
    assert not missing, f"overlay signers missing: {missing}"


# ---- Activation -----------------------------------------------------------
def test_forbidden_criteria_has_all_five(strict_adult_overlay):
    forbidden = set(
        strict_adult_overlay["activation"]["forbidden_criteria"]
    )
    missing = REQUIRED_FORBIDDEN_CRITERIA - forbidden
    assert not missing, f"overlay forbidden_criteria missing: {missing}"


def test_activation_criteria_present(strict_adult_overlay):
    criteria = strict_adult_overlay["activation"]["criteria"]
    assert isinstance(criteria, list) and len(criteria) >= 1


# ---- Overrides: category 10 severity floor 5 ------------------------------
def test_category_10_override_has_severity_floor_5(strict_adult_overlay):
    overrides = strict_adult_overlay["overrides"]
    cat10 = [o for o in overrides if o.get("category") == 10]
    assert cat10, "overlay must define an override for category 10"
    assert len(cat10) == 1, "exactly one override expected for category 10"
    assert cat10[0]["severity_floor"] == 5, (
        "category 10 (SEXUAL_ADULT) must have severity_floor 5 "
        "(jurisdictional ban)"
    )


def test_no_override_relaxes_child_safety_floor(strict_adult_overlay):
    for override in strict_adult_overlay["overrides"]:
        if override.get("category") == 1:
            assert override["severity_floor"] >= 5, (
                "CHILD_SAFETY floor cannot be lowered below 5"
            )


# ---- Allowed contexts -----------------------------------------------------
def test_allowed_contexts_include_protected_speech(strict_adult_overlay):
    contexts = set(strict_adult_overlay["allowed_contexts"])
    missing = REQUIRED_ALLOWED_CONTEXTS - contexts
    assert not missing, f"allowed_contexts missing: {missing}"


# ---- Expiry ---------------------------------------------------------------
def test_expires_on_is_within_18_months(strict_adult_overlay):
    expires = _as_date(strict_adult_overlay["expires_on"])
    today = date.today()
    assert expires > today, "expires_on must be in the future"
    # 18 months max; allow a small tolerance for leap / month-length
    # drift — 18 * 31 days is generous.
    delta_days = (expires - today).days
    assert delta_days <= 18 * 31, (
        f"expires_on must be <= ~18 months from today; got {delta_days} days"
    )


# ---- User notice ----------------------------------------------------------
def test_user_notice_has_summary(strict_adult_overlay):
    notice = strict_adult_overlay["user_notice"]
    assert isinstance(notice.get("visible_pack_summary"), str)
    assert notice["visible_pack_summary"].strip()
    assert "appeal_resource_id" in notice
    assert "opt_out_allowed" in notice


# ---- Local language assets / normalization ------------------------------
def test_local_language_assets_have_normalization(strict_adult_overlay):
    norm = strict_adult_overlay["local_language_assets"]["normalization"]
    assert norm["nfkc"] is True
    assert norm["case_fold"] is True
    assert norm["homoglyph_map_id"]
    assert isinstance(norm["transliteration_refs"], list)


def test_normalization_file_matches_overlay(
    strict_adult_normalization, strict_adult_overlay
):
    overlay_norm = strict_adult_overlay["local_language_assets"]["normalization"]
    assert strict_adult_normalization["nfkc"] is True
    assert strict_adult_normalization["case_fold"] is True
    assert (
        strict_adult_normalization["homoglyph_map_id"]
        == overlay_norm["homoglyph_map_id"]
    )


# ---- Directory structure --------------------------------------------------
def test_lexicons_directory_exists(jurisdictions_dir):
    lexicons = jurisdictions_dir / "archetype-strict-adult" / "lexicons"
    assert lexicons.is_dir(), (
        "archetype-strict-adult must ship a lexicons/ subdirectory"
    )
