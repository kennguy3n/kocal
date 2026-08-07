"""Validate the jurisdiction overlay template.

Template lives at ``kchat-skills/jurisdictions/_template/overlay.yaml``
and encodes the shape required by ARCHITECTURE.md "Jurisdiction Overlay
Template" (lines 421-488).
"""
from __future__ import annotations

from pathlib import Path

import pytest

TEMPLATE_PATH = (
    Path(__file__).resolve().parents[2]
    / "jurisdictions"
    / "_template"
    / "overlay.yaml"
)


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

REQUIRED_SIGNERS = {"legal_review", "cultural_review"}

REQUIRED_ALLOWED_CONTEXTS = {
    "QUOTED_SPEECH_CONTEXT",
    "NEWS_CONTEXT",
    "EDUCATION_CONTEXT",
    "COUNTERSPEECH_CONTEXT",
}

REQUIRED_LOCAL_DEFINITIONS = {
    "legal_age_general",
    "legal_age_sexual_content_consumer",
    "legal_age_marketplace_alcohol",
    "legal_age_marketplace_tobacco",
    "protected_classes",
    "listed_extremist_orgs",
    "restricted_symbols",
    "election_rules",
}

REQUIRED_LOCAL_LANGUAGE_ASSETS = {
    "primary_languages",
    "lexicons",
    "normalization",
}

REQUIRED_NORMALIZATION_FIELDS = {
    "nfkc",
    "case_fold",
    "homoglyph_map_id",
    "transliteration_refs",
}

REQUIRED_USER_NOTICE = {
    "visible_pack_summary",
    "appeal_resource_id",
    "opt_out_allowed",
}


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------
def test_template_parses_as_valid_yaml(jurisdiction_template):
    assert isinstance(jurisdiction_template, dict)


def test_template_required_top_level_keys(jurisdiction_template):
    missing = REQUIRED_TOP_LEVEL - set(jurisdiction_template.keys())
    assert not missing, f"template missing top-level keys: {missing}"


def test_template_parent_is_global_baseline(jurisdiction_template):
    assert jurisdiction_template["parent"] == "kchat.global.guardrail.baseline"


def test_template_schema_version_is_1(jurisdiction_template):
    assert jurisdiction_template["schema_version"] == 1


def test_template_signers_include_legal_and_cultural(jurisdiction_template):
    signers = set(jurisdiction_template["signers"])
    missing = REQUIRED_SIGNERS - signers
    assert not missing, f"template signers missing: {missing}"
    assert "trust_and_safety" in signers, (
        "trust_and_safety must always sign a jurisdiction pack"
    )


def test_template_skill_id_pattern(jurisdiction_template):
    skill_id = jurisdiction_template["skill_id"]
    assert skill_id.startswith("kchat.jurisdiction.")
    assert skill_id.endswith(".guardrail.v1")


# ---- Activation -----------------------------------------------------------
def test_template_forbidden_criteria_has_all_five(jurisdiction_template):
    forbidden = set(jurisdiction_template["activation"]["forbidden_criteria"])
    missing = REQUIRED_FORBIDDEN_CRITERIA - forbidden
    assert not missing, (
        f"template forbidden_criteria missing: {missing}. "
        "All 5 forbidden activation methods must be listed."
    )


def test_template_activation_criteria_present(jurisdiction_template):
    criteria = jurisdiction_template["activation"]["criteria"]
    assert isinstance(criteria, list) and len(criteria) >= 1


# ---- Local definitions ---------------------------------------------------
def test_template_local_definitions_required_keys(jurisdiction_template):
    defs = jurisdiction_template["local_definitions"]
    missing = REQUIRED_LOCAL_DEFINITIONS - set(defs.keys())
    assert not missing, f"template local_definitions missing: {missing}"


# ---- Language assets -----------------------------------------------------
def test_template_language_assets_keys(jurisdiction_template):
    assets = jurisdiction_template["local_language_assets"]
    missing = REQUIRED_LOCAL_LANGUAGE_ASSETS - set(assets.keys())
    assert not missing


def test_template_normalization_required_fields(jurisdiction_template):
    norm = jurisdiction_template["local_language_assets"]["normalization"]
    missing = REQUIRED_NORMALIZATION_FIELDS - set(norm.keys())
    assert not missing, f"template normalization missing: {missing}"
    assert norm["nfkc"] is True
    assert norm["case_fold"] is True


# ---- Overrides / allowed_contexts / user_notice --------------------------
def test_template_has_at_least_one_override(jurisdiction_template):
    assert len(jurisdiction_template["overrides"]) >= 1


def test_template_allowed_contexts_match_protected_speech(jurisdiction_template):
    ctx = set(jurisdiction_template["allowed_contexts"])
    assert ctx == REQUIRED_ALLOWED_CONTEXTS


def test_template_user_notice_required_fields(jurisdiction_template):
    notice = jurisdiction_template["user_notice"]
    missing = REQUIRED_USER_NOTICE - set(notice.keys())
    assert not missing


# ---- Raw-text regression: forbidden criteria verbatim --------------------
@pytest.mark.parametrize(
    "forbidden", sorted(REQUIRED_FORBIDDEN_CRITERIA)
)
def test_template_source_mentions_forbidden_criterion_verbatim(forbidden):
    text = TEMPLATE_PATH.read_text(encoding="utf-8")
    assert forbidden in text, (
        f"forbidden criterion {forbidden!r} must appear verbatim in the template"
    )
