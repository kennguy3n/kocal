"""Validate the kchat.jurisdiction.se.guardrail.v1 country pack.

Country-specific overrides match
the overlay's docstring:
  Brottsbalk child protection (Chapter 16 §10a); Terrorbrottslag (2003:148).
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "se"


def test_run_all_structural_assertions(se_overlay, se_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        se_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=se_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(se_overlay):
    override = A.get_override(se_overlay, 1)
    assert override["severity_floor"] == 5, (
        "SE: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_4(se_overlay):
    override = A.get_override(se_overlay, 4)
    assert override["severity_floor"] == 4, (
        "SE: category 4 override must be severity_floor 4"
    )


def test_legal_age_marketplace_alcohol_is_20(se_overlay):
    assert se_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 20


def test_legal_age_marketplace_tobacco_is_18(se_overlay):
    assert se_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 18


def test_protected_classes_includes_statutory_enumeration(se_overlay):
    classes = set(se_overlay["local_definitions"]["protected_classes"])
    expected = {'sex', 'disability', 'ethnic_origin', 'religion_or_belief'}
    assert expected <= classes, (
        "SE protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(se_overlay):
    langs = list(se_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['sv'], (
        f"SE primary_languages must equal ['sv']; got {langs}"
    )


def test_user_notice_opt_out_allowed_true(se_overlay):
    assert se_overlay["user_notice"]["opt_out_allowed"] is True


def test_election_rules_reference_authority(se_overlay):
    rules = se_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "se_valmyndigheten_authority_v1"


def test_pack_passes_anti_misuse_validation(se_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(se_overlay)
    assert report.passed, f"SE pack failed anti-misuse: {report.errors}"
