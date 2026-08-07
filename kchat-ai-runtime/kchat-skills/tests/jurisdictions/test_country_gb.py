"""Validate the kchat.jurisdiction.gb.guardrail.v1 country pack.

Country-specific overrides match
the overlay's docstring:
  Online Safety Act 2023; Terrorism Act 2000; child protection (Protection of Children Act 1978).
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "gb"


def test_run_all_structural_assertions(gb_overlay, gb_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        gb_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=gb_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(gb_overlay):
    override = A.get_override(gb_overlay, 1)
    assert override["severity_floor"] == 5, (
        "GB: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_4(gb_overlay):
    override = A.get_override(gb_overlay, 4)
    assert override["severity_floor"] == 4, (
        "GB: category 4 override must be severity_floor 4"
    )


def test_category_7_scam_fraud_floor_3(gb_overlay):
    override = A.get_override(gb_overlay, 7)
    assert override["severity_floor"] == 3, (
        "GB: category 7 override must be severity_floor 3"
    )


def test_legal_age_marketplace_alcohol_is_18(gb_overlay):
    assert gb_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 18


def test_legal_age_marketplace_tobacco_is_18(gb_overlay):
    assert gb_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 18


def test_protected_classes_includes_statutory_enumeration(gb_overlay):
    classes = set(gb_overlay["local_definitions"]["protected_classes"])
    expected = {'sex', 'race', 'religion_or_belief', 'disability', 'age'}
    assert expected <= classes, (
        "GB protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(gb_overlay):
    langs = list(gb_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['en'], (
        f"GB primary_languages must equal ['en']; got {langs}"
    )


def test_user_notice_opt_out_allowed_true(gb_overlay):
    assert gb_overlay["user_notice"]["opt_out_allowed"] is True


def test_election_rules_reference_authority(gb_overlay):
    rules = gb_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "gb_electoral_commission_authority_v1"


def test_pack_passes_anti_misuse_validation(gb_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(gb_overlay)
    assert report.passed, f"GB pack failed anti-misuse: {report.errors}"
