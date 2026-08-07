"""Validate the kchat.jurisdiction.hu.guardrail.v1 country pack.

Country-specific overrides
match the overlay's docstring:
  Btk. 204 child protection; Btk. 314-318 terrorism offences.
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "hu"


def test_run_all_structural_assertions(hu_overlay, hu_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        hu_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=hu_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(hu_overlay):
    override = A.get_override(hu_overlay, 1)
    assert override["severity_floor"] == 5, (
        "HU: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_4(hu_overlay):
    override = A.get_override(hu_overlay, 4)
    assert override["severity_floor"] == 4, (
        "HU: category 4 override must be severity_floor 4"
    )


def test_legal_age_marketplace_alcohol_is_18(hu_overlay):
    assert hu_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 18


def test_legal_age_marketplace_tobacco_is_18(hu_overlay):
    assert hu_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 18


def test_protected_classes_includes_statutory_enumeration(hu_overlay):
    classes = set(hu_overlay["local_definitions"]["protected_classes"])
    expected = {'ethnic_origin', 'sex', 'religion', 'race'}
    assert expected <= classes, (
        "HU protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(hu_overlay):
    langs = list(hu_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['hu'], (
        f"HU primary_languages must equal ['hu']; got {langs}"
    )


def test_user_notice_opt_out_allowed_true(hu_overlay):
    assert hu_overlay["user_notice"]["opt_out_allowed"] is True


def test_election_rules_reference_authority(hu_overlay):
    rules = hu_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "hu_nvi_authority_v1"


def test_pack_passes_anti_misuse_validation(hu_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(hu_overlay)
    assert report.passed, f"HU pack failed anti-misuse: {report.errors}"
