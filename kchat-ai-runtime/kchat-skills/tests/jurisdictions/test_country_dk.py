"""Validate the kchat.jurisdiction.dk.guardrail.v1 country pack.

Country-specific overrides
match the overlay's docstring:
  Straffeloven 235 child protection; Straffeloven 114 terrorism.
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "dk"


def test_run_all_structural_assertions(dk_overlay, dk_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        dk_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=dk_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(dk_overlay):
    override = A.get_override(dk_overlay, 1)
    assert override["severity_floor"] == 5, (
        "DK: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_4(dk_overlay):
    override = A.get_override(dk_overlay, 4)
    assert override["severity_floor"] == 4, (
        "DK: category 4 override must be severity_floor 4"
    )


def test_legal_age_marketplace_alcohol_is_18(dk_overlay):
    assert dk_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 18


def test_legal_age_marketplace_tobacco_is_18(dk_overlay):
    assert dk_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 18


def test_protected_classes_includes_statutory_enumeration(dk_overlay):
    classes = set(dk_overlay["local_definitions"]["protected_classes"])
    expected = {'ethnic_origin', 'sex', 'religion', 'race'}
    assert expected <= classes, (
        "DK protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(dk_overlay):
    langs = list(dk_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['da'], (
        f"DK primary_languages must equal ['da']; got {langs}"
    )


def test_user_notice_opt_out_allowed_true(dk_overlay):
    assert dk_overlay["user_notice"]["opt_out_allowed"] is True


def test_election_rules_reference_authority(dk_overlay):
    rules = dk_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "dk_im_authority_v1"


def test_pack_passes_anti_misuse_validation(dk_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(dk_overlay)
    assert report.passed, f"DK pack failed anti-misuse: {report.errors}"
