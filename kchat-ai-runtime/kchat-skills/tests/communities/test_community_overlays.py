"""Validation tests for the 38 community overlays in kchat-skills/communities/.

8 first-wave overlays (school, family, workplace, adult_only, marketplace,
health_support, political, gaming) plus 30 expansion overlays.

See ARCHITECTURE.md "Community Overlay Template" (lines 491-535).
"""
from __future__ import annotations

import pytest
import yaml

from .conftest import COMMUNITIES_DIR, COMMUNITY_FILES

REQUIRED_KEYS = {
    "skill_id",
    "parent",
    "schema_version",
    "signers",
    "community_profile",
    "rules",
}
REQUIRED_PROFILE_KEYS = {"kind", "age_mode", "visibility", "set_by"}
VALID_AGE_MODES = {"minor_present", "mixed_age", "adult_only"}
VALID_ACTIONS = {"label_only", "warn", "strong_warn", "block"}


@pytest.mark.parametrize("name", COMMUNITY_FILES)
def test_community_yaml_is_valid(name):
    path = COMMUNITIES_DIR / name
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    assert isinstance(loaded, dict), f"{name} must parse to a mapping"


@pytest.mark.parametrize("name", COMMUNITY_FILES)
def test_community_required_keys(community_overlays, name):
    overlay = community_overlays[name]
    missing = REQUIRED_KEYS - set(overlay.keys())
    assert not missing, f"{name} missing required keys: {missing}"


@pytest.mark.parametrize("name", COMMUNITY_FILES)
def test_community_parent_is_global_baseline(community_overlays, name):
    assert community_overlays[name]["parent"] == "kchat.global.guardrail.baseline"


@pytest.mark.parametrize("name", COMMUNITY_FILES)
def test_community_schema_version_is_1(community_overlays, name):
    assert community_overlays[name]["schema_version"] == 1


@pytest.mark.parametrize("name", COMMUNITY_FILES)
def test_community_signers_includes_trust_and_safety(community_overlays, name):
    signers = community_overlays[name]["signers"]
    assert isinstance(signers, list)
    assert "trust_and_safety" in signers


@pytest.mark.parametrize("name", COMMUNITY_FILES)
def test_community_profile_required_fields(community_overlays, name):
    profile = community_overlays[name]["community_profile"]
    missing = REQUIRED_PROFILE_KEYS - set(profile.keys())
    assert not missing, f"{name} community_profile missing: {missing}"


@pytest.mark.parametrize("name", COMMUNITY_FILES)
def test_community_age_mode_valid(community_overlays, name):
    age_mode = community_overlays[name]["community_profile"]["age_mode"]
    assert age_mode in VALID_AGE_MODES, (
        f"{name} has invalid age_mode {age_mode!r}"
    )


@pytest.mark.parametrize("name", COMMUNITY_FILES)
def test_community_skill_id_format(community_overlays, name):
    skill_id = community_overlays[name]["skill_id"]
    assert skill_id.startswith("kchat.community.")
    assert skill_id.endswith(".guardrail.v1")


@pytest.mark.parametrize("name", COMMUNITY_FILES)
def test_community_rules_categories_in_taxonomy_range(community_overlays, name):
    for rule in community_overlays[name]["rules"]:
        cat = rule.get("category")
        assert isinstance(cat, int), f"{name}: rule.category must be int"
        assert 0 <= cat <= 16, (
            f"{name}: rule.category={cat} outside 0..16 taxonomy range"
        )


@pytest.mark.parametrize("name", COMMUNITY_FILES)
def test_community_rule_actions_valid(community_overlays, name):
    for rule in community_overlays[name]["rules"]:
        # rule_set entries (community-rule sub-rules) carry their own
        # action; otherwise the top-level rule has an `action`.
        if "action" in rule:
            assert rule["action"] in VALID_ACTIONS, (
                f"{name}: invalid action {rule['action']!r}"
            )
        if "rule_set" in rule:
            for sub in rule["rule_set"]:
                assert sub["action"] in VALID_ACTIONS


# --- Specific overlays ----------------------------------------------------


def _find_rule(overlay: dict, category: int) -> dict | None:
    for r in overlay["rules"]:
        if r.get("category") == category:
            return r
    return None


def test_school_age_mode_minor_present_and_blocks_sexual_adult(community_overlays):
    school = community_overlays["school.yaml"]
    assert school["community_profile"]["age_mode"] == "minor_present"
    sexual_rule = _find_rule(school, 10)
    assert sexual_rule is not None, "school.yaml must define a SEXUAL_ADULT rule"
    assert sexual_rule["action"] == "block"


def test_adult_only_age_mode(community_overlays):
    adult_only = community_overlays["adult_only.yaml"]
    assert adult_only["community_profile"]["age_mode"] == "adult_only"


def test_health_support_loosens_self_harm(community_overlays):
    health = community_overlays["health_support.yaml"]
    self_harm = _find_rule(health, 2)
    assert self_harm is not None
    assert self_harm["action"] == "label_only"


def test_workplace_has_scam_links_counter(community_overlays):
    workplace = community_overlays["workplace.yaml"]
    counters = workplace.get("group_risk_counters", [])
    counter_ids = {c["counter_id"] for c in counters}
    assert "group_scam_links_24h" in counter_ids


def test_marketplace_tightens_scam_and_illegal_goods(community_overlays):
    marketplace = community_overlays["marketplace.yaml"]
    scam = _find_rule(marketplace, 7)
    illegal = _find_rule(marketplace, 12)
    drugs = _find_rule(marketplace, 11)
    assert scam is not None and scam["action"] == "strong_warn"
    assert illegal is not None and illegal["action"] == "strong_warn"
    assert drugs is not None and drugs["action"] == "strong_warn"


def test_political_tightens_civic_misinfo(community_overlays):
    political = community_overlays["political.yaml"]
    assert political["community_profile"]["age_mode"] == "adult_only"
    civic = _find_rule(political, 14)
    assert civic is not None and civic["action"] == "warn"


def test_gaming_has_violence_threats_counter(community_overlays):
    gaming = community_overlays["gaming.yaml"]
    counters = gaming.get("group_risk_counters", [])
    counter_ids = {c["counter_id"] for c in counters}
    assert "group_violence_threats_7d" in counter_ids


def test_family_age_mode_mixed_and_strong_warn_sexual_adult(community_overlays):
    family = community_overlays["family.yaml"]
    assert family["community_profile"]["age_mode"] == "mixed_age"
    sex = _find_rule(family, 10)
    assert sex is not None and sex["action"] == "strong_warn"


# --- Expansion overlays --------------------------------------------------


def test_dating_age_mode_adult_only(community_overlays):
    dating = community_overlays["dating.yaml"]
    assert dating["community_profile"]["age_mode"] == "adult_only"
    sex = _find_rule(dating, 10)
    assert sex is not None and sex["action"] == "label_only"
    scam = _find_rule(dating, 7)
    assert scam is not None and scam["action"] == "strong_warn"


def test_mental_health_loosens_self_harm_for_peer_support(community_overlays):
    mh = community_overlays["mental_health.yaml"]
    assert mh["community_profile"]["age_mode"] == "adult_only"
    sh = _find_rule(mh, 2)
    assert sh is not None and sh["action"] == "label_only", (
        "mental_health peer-support context loosens SELF_HARM (cat 2) "
        "to label_only — mirrors the existing health_support pattern."
    )


def test_journalism_loosens_extremism_for_news_context(community_overlays):
    j = community_overlays["journalism.yaml"]
    assert j["community_profile"]["age_mode"] == "adult_only"
    ext = _find_rule(j, 4)
    assert ext is not None and ext["action"] == "label_only", (
        "journalism overlay relies on the NEWS_CONTEXT carve-out and "
        "loosens EXTREMISM (cat 4) to a label so news coverage is "
        "preserved."
    )


def test_seniors_tightens_scam(community_overlays):
    s = community_overlays["seniors.yaml"]
    assert s["community_profile"]["age_mode"] == "adult_only"
    scam = _find_rule(s, 7)
    assert scam is not None and scam["action"] == "strong_warn"
    pii = _find_rule(s, 9)
    assert pii is not None and pii["action"] == "strong_warn"


def test_religious_tightens_hate(community_overlays):
    rel = community_overlays["religious.yaml"]
    hate = _find_rule(rel, 6)
    assert hate is not None and hate["action"] == "strong_warn"


def test_lgbtq_support_strengthens_hate_and_harassment(community_overlays):
    lgbtq = community_overlays["lgbtq_support.yaml"]
    assert lgbtq["community_profile"]["age_mode"] == "adult_only"
    hate = _find_rule(lgbtq, 6)
    assert hate is not None and hate["action"] == "strong_warn"
    har = _find_rule(lgbtq, 5)
    assert har is not None and har["action"] == "strong_warn"


def test_emergency_response_tightens_health_misinformation(community_overlays):
    er = community_overlays["emergency_response.yaml"]
    health = _find_rule(er, 13)
    assert health is not None and health["action"] == "strong_warn"


def test_phase6_expansion_overlays_pass_anti_misuse_validation(
    community_overlays,
):
    """Every expansion overlay must pass anti_misuse.validate_pack."""
    from compiler.anti_misuse import validate_pack

    phase6 = [
        n for n in community_overlays if n not in {
            "school.yaml", "family.yaml", "workplace.yaml",
            "adult_only.yaml", "marketplace.yaml", "health_support.yaml",
            "political.yaml", "gaming.yaml",
        }
    ]
    assert len(phase6) == 30, (
        f"expected 30 expansion overlays; got {len(phase6)}"
    )
    for name in phase6:
        report = validate_pack(community_overlays[name])
        assert report.passed, (
            f"{name} failed anti-misuse validation: {report.errors}"
        )


def test_total_community_overlay_count(community_overlays):
    """Coverage target: 38 community overlays total (8 + 30)."""
    assert len(community_overlays) == 38, (
        f"expected 38 community overlays (8 first-wave + 30 expansion); "
        f"got {len(community_overlays)}"
    )


# --- Template -------------------------------------------------------------


def test_template_overlay_has_required_keys(community_template):
    missing = REQUIRED_KEYS - set(community_template.keys())
    assert not missing, f"template overlay.yaml missing: {missing}"
