"""Validate the kchat.jurisdiction.il.guardrail.v1 country pack.

Country-specific overrides
match the overlay's docstring:
  Penal Code 214 child protection; Counter-Terrorism Law 5776-2016.
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "il"


def test_run_all_structural_assertions(il_overlay, il_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        il_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=il_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(il_overlay):
    override = A.get_override(il_overlay, 1)
    assert override["severity_floor"] == 5, (
        "IL: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_4(il_overlay):
    override = A.get_override(il_overlay, 4)
    assert override["severity_floor"] == 4, (
        "IL: category 4 override must be severity_floor 4"
    )


def test_legal_age_marketplace_alcohol_is_18(il_overlay):
    assert il_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 18


def test_legal_age_marketplace_tobacco_is_18(il_overlay):
    assert il_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 18


def test_protected_classes_includes_statutory_enumeration(il_overlay):
    classes = set(il_overlay["local_definitions"]["protected_classes"])
    expected = {'ethnic_origin', 'sex', 'religion', 'race'}
    assert expected <= classes, (
        "IL protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(il_overlay):
    langs = list(il_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['he', 'ar'], (
        f"IL primary_languages must equal ['he', 'ar']; got {langs}"
    )


def test_user_notice_opt_out_allowed_true(il_overlay):
    assert il_overlay["user_notice"]["opt_out_allowed"] is True


def test_election_rules_reference_authority(il_overlay):
    rules = il_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "il_central_elections_authority_v1"


def test_pack_passes_anti_misuse_validation(il_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(il_overlay)
    assert report.passed, f"IL pack failed anti-misuse: {report.errors}"
