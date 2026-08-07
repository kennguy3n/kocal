"""Validate the kchat.jurisdiction.ma.guardrail.v1 country pack.

Country-specific overrides
match the overlay's docstring:
  Code Penal Art. 503 child protection; Loi 03-03 anti-terrorism.
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "ma"


def test_run_all_structural_assertions(ma_overlay, ma_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        ma_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=ma_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(ma_overlay):
    override = A.get_override(ma_overlay, 1)
    assert override["severity_floor"] == 5, (
        "MA: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_4(ma_overlay):
    override = A.get_override(ma_overlay, 4)
    assert override["severity_floor"] == 4, (
        "MA: category 4 override must be severity_floor 4"
    )


def test_legal_age_marketplace_alcohol_is_18(ma_overlay):
    assert ma_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 18


def test_legal_age_marketplace_tobacco_is_18(ma_overlay):
    assert ma_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 18


def test_protected_classes_includes_statutory_enumeration(ma_overlay):
    classes = set(ma_overlay["local_definitions"]["protected_classes"])
    expected = {'ethnic_origin', 'sex', 'religion', 'race'}
    assert expected <= classes, (
        "MA protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(ma_overlay):
    langs = list(ma_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['ar', 'fr'], (
        f"MA primary_languages must equal ['ar', 'fr']; got {langs}"
    )


def test_user_notice_opt_out_allowed_true(ma_overlay):
    assert ma_overlay["user_notice"]["opt_out_allowed"] is True


def test_election_rules_reference_authority(ma_overlay):
    rules = ma_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "ma_cndh_authority_v1"


def test_pack_passes_anti_misuse_validation(ma_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(ma_overlay)
    assert report.passed, f"MA pack failed anti-misuse: {report.errors}"
