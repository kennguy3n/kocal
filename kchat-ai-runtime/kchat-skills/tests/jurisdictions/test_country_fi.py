"""Validate the kchat.jurisdiction.fi.guardrail.v1 country pack.

Country-specific overrides
match the overlay's docstring:
  Rikoslaki 17:18-19 child protection; 34a luku terrorism.
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "fi"


def test_run_all_structural_assertions(fi_overlay, fi_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        fi_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=fi_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(fi_overlay):
    override = A.get_override(fi_overlay, 1)
    assert override["severity_floor"] == 5, (
        "FI: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_4(fi_overlay):
    override = A.get_override(fi_overlay, 4)
    assert override["severity_floor"] == 4, (
        "FI: category 4 override must be severity_floor 4"
    )


def test_legal_age_marketplace_alcohol_is_18(fi_overlay):
    assert fi_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 18


def test_legal_age_marketplace_tobacco_is_18(fi_overlay):
    assert fi_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 18


def test_protected_classes_includes_statutory_enumeration(fi_overlay):
    classes = set(fi_overlay["local_definitions"]["protected_classes"])
    expected = {'ethnic_origin', 'sex', 'religion', 'race'}
    assert expected <= classes, (
        "FI protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(fi_overlay):
    langs = list(fi_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['fi'], (
        f"FI primary_languages must equal ['fi']; got {langs}"
    )


def test_user_notice_opt_out_allowed_true(fi_overlay):
    assert fi_overlay["user_notice"]["opt_out_allowed"] is True


def test_election_rules_reference_authority(fi_overlay):
    rules = fi_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "fi_oikeusministerio_authority_v1"


def test_pack_passes_anti_misuse_validation(fi_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(fi_overlay)
    assert report.passed, f"FI pack failed anti-misuse: {report.errors}"
