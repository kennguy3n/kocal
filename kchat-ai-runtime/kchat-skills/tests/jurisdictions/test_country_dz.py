"""Validate the kchat.jurisdiction.dz.guardrail.v1 country pack.

Country-specific overrides
match the overlay's docstring:
  Loi 15-12 child protection; Code Penal Art. 87 bis anti-terrorism.
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "dz"


def test_run_all_structural_assertions(dz_overlay, dz_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        dz_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=dz_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(dz_overlay):
    override = A.get_override(dz_overlay, 1)
    assert override["severity_floor"] == 5, (
        "DZ: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_4(dz_overlay):
    override = A.get_override(dz_overlay, 4)
    assert override["severity_floor"] == 4, (
        "DZ: category 4 override must be severity_floor 4"
    )


def test_legal_age_marketplace_alcohol_is_18(dz_overlay):
    assert dz_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 18


def test_legal_age_marketplace_tobacco_is_18(dz_overlay):
    assert dz_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 18


def test_protected_classes_includes_statutory_enumeration(dz_overlay):
    classes = set(dz_overlay["local_definitions"]["protected_classes"])
    expected = {'ethnic_origin', 'sex', 'religion', 'race'}
    assert expected <= classes, (
        "DZ protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(dz_overlay):
    langs = list(dz_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['ar', 'fr'], (
        f"DZ primary_languages must equal ['ar', 'fr']; got {langs}"
    )


def test_user_notice_opt_out_allowed_true(dz_overlay):
    assert dz_overlay["user_notice"]["opt_out_allowed"] is True


def test_election_rules_reference_authority(dz_overlay):
    rules = dz_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "dz_anie_authority_v1"


def test_pack_passes_anti_misuse_validation(dz_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(dz_overlay)
    assert report.passed, f"DZ pack failed anti-misuse: {report.errors}"
