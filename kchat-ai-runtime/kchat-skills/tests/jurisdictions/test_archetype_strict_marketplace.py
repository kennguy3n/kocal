"""Validate the kchat.jurisdiction.archetype-strict-marketplace pack.

The pack defines the strict marketplace / restricted-goods archetype:
severity floor 4 on categories 11 and 12. See ARCHITECTURE.md
"Jurisdiction Overlay Template" for the compiled-prompt section
this archetype produces.
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
def test_parses_as_valid_yaml(strict_marketplace_overlay):
    assert isinstance(strict_marketplace_overlay, dict)


def test_required_top_level_keys(strict_marketplace_overlay):
    missing = REQUIRED_TOP_LEVEL - set(strict_marketplace_overlay.keys())
    assert not missing, f"overlay missing top-level keys: {missing}"


def test_skill_id_matches_pattern(strict_marketplace_overlay):
    skill_id = strict_marketplace_overlay["skill_id"]
    assert skill_id == (
        "kchat.jurisdiction.archetype-strict-marketplace.guardrail.v1"
    )
    assert skill_id.startswith("kchat.jurisdiction.")
    assert skill_id.endswith(".guardrail.v1")


def test_parent_is_global_baseline(strict_marketplace_overlay):
    assert (
        strict_marketplace_overlay["parent"]
        == "kchat.global.guardrail.baseline"
    )


def test_schema_version_is_1(strict_marketplace_overlay):
    assert strict_marketplace_overlay["schema_version"] == 1


# ---- Signers --------------------------------------------------------------
def test_signers_include_legal_and_cultural_review(strict_marketplace_overlay):
    signers = set(strict_marketplace_overlay["signers"])
    missing = REQUIRED_SIGNERS - signers
    assert not missing, f"overlay signers missing: {missing}"


# ---- Activation -----------------------------------------------------------
def test_forbidden_criteria_has_all_five(strict_marketplace_overlay):
    forbidden = set(
        strict_marketplace_overlay["activation"]["forbidden_criteria"]
    )
    missing = REQUIRED_FORBIDDEN_CRITERIA - forbidden
    assert not missing, f"overlay forbidden_criteria missing: {missing}"


def test_activation_uses_archetype_region_code(strict_marketplace_overlay):
    criteria = strict_marketplace_overlay["activation"]["criteria"]
    # Each criterion is a single-key mapping; the value must be the
    # archetype region code.
    values = []
    for c in criteria:
        assert isinstance(c, dict) and len(c) == 1
        values.append(next(iter(c.values())))
    assert all(v == "archetype-strict-marketplace" for v in values), (
        "every activation criterion must use the archetype region code"
    )


# ---- Overrides: cat 11 floor 4, cat 12 floor 4 ---------------------------
def test_category_11_override_has_severity_floor_4(strict_marketplace_overlay):
    override = _get_override(strict_marketplace_overlay, 11)
    assert override["severity_floor"] == 4, (
        "category 11 (DRUGS_WEAPONS) must have severity_floor 4"
    )


def test_category_12_override_has_severity_floor_4(strict_marketplace_overlay):
    override = _get_override(strict_marketplace_overlay, 12)
    assert override["severity_floor"] == 4, (
        "category 12 (ILLEGAL_GOODS) must have severity_floor 4"
    )


def test_no_override_relaxes_child_safety_floor(strict_marketplace_overlay):
    for override in strict_marketplace_overlay["overrides"]:
        if override.get("category") == 1:
            assert override["severity_floor"] >= 5, (
                "CHILD_SAFETY floor cannot be lowered below 5"
            )


# ---- Allowed contexts -----------------------------------------------------
def test_allowed_contexts_include_all_four_protected_speech_contexts(
    strict_marketplace_overlay,
):
    contexts = set(strict_marketplace_overlay["allowed_contexts"])
    missing = REQUIRED_ALLOWED_CONTEXTS - contexts
    assert not missing, (
        "archetype-strict-marketplace must declare all 4 protected-speech "
        f"contexts; missing: {missing}"
    )


# ---- Expiry ---------------------------------------------------------------
def test_expires_on_is_future_and_within_18_months(strict_marketplace_overlay):
    expires = _as_date(strict_marketplace_overlay["expires_on"])
    today = date.today()
    assert expires > today, "expires_on must be a valid future date"
    assert (expires - today).days <= 18 * 31


# ---- User notice ----------------------------------------------------------
def test_user_notice_has_summary(strict_marketplace_overlay):
    notice = strict_marketplace_overlay["user_notice"]
    assert isinstance(notice.get("visible_pack_summary"), str)
    assert notice["visible_pack_summary"].strip()
    assert "appeal_resource_id" in notice
    assert "opt_out_allowed" in notice


# ---- Local language assets / normalization ------------------------------
def test_local_language_assets_have_normalization(strict_marketplace_overlay):
    norm = strict_marketplace_overlay["local_language_assets"]["normalization"]
    assert norm["nfkc"] is True
    assert norm["case_fold"] is True
    assert norm["homoglyph_map_id"]
    assert isinstance(norm["transliteration_refs"], list)


def test_normalization_file_has_all_four_required_fields(
    strict_marketplace_normalization,
):
    assert strict_marketplace_normalization["nfkc"] is True
    assert strict_marketplace_normalization["case_fold"] is True
    assert strict_marketplace_normalization["homoglyph_map_id"]
    assert isinstance(
        strict_marketplace_normalization["transliteration_refs"], list
    )


def test_normalization_file_matches_overlay(
    strict_marketplace_normalization, strict_marketplace_overlay
):
    overlay_norm = strict_marketplace_overlay[
        "local_language_assets"
    ]["normalization"]
    assert (
        strict_marketplace_normalization["homoglyph_map_id"]
        == overlay_norm["homoglyph_map_id"]
    )


# ---- Directory structure --------------------------------------------------
def test_lexicons_directory_exists(jurisdictions_dir):
    lexicons = (
        jurisdictions_dir / "archetype-strict-marketplace" / "lexicons"
    )
    assert lexicons.is_dir(), (
        "archetype-strict-marketplace must ship a lexicons/ subdirectory"
    )
