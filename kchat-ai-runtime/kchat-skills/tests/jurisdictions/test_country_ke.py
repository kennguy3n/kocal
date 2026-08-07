"""Validate the kchat.jurisdiction.ke.guardrail.v1 country pack.

Country-specific overrides match
the overlay's docstring:
  Children Act 2022; Prevention of Terrorism Act 2012.
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "ke"


def test_run_all_structural_assertions(ke_overlay, ke_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        ke_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=ke_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(ke_overlay):
    override = A.get_override(ke_overlay, 1)
    assert override["severity_floor"] == 5, (
        "KE: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_4(ke_overlay):
    override = A.get_override(ke_overlay, 4)
    assert override["severity_floor"] == 4, (
        "KE: category 4 override must be severity_floor 4"
    )


def test_legal_age_marketplace_alcohol_is_18(ke_overlay):
    assert ke_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 18


def test_legal_age_marketplace_tobacco_is_18(ke_overlay):
    assert ke_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 18


def test_protected_classes_includes_statutory_enumeration(ke_overlay):
    classes = set(ke_overlay["local_definitions"]["protected_classes"])
    expected = {'religion', 'sex', 'disability', 'race'}
    assert expected <= classes, (
        "KE protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(ke_overlay):
    langs = list(ke_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['en', 'sw'], (
        f"KE primary_languages must equal ['en', 'sw']; got {langs}"
    )


def test_user_notice_opt_out_allowed_true(ke_overlay):
    assert ke_overlay["user_notice"]["opt_out_allowed"] is True


def test_election_rules_reference_authority(ke_overlay):
    rules = ke_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "ke_iebc_authority_v1"


def test_pack_passes_anti_misuse_validation(ke_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(ke_overlay)
    assert report.passed, f"KE pack failed anti-misuse: {report.errors}"
