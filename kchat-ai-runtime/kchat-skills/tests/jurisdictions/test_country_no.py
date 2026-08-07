"""Validate the kchat.jurisdiction.no.guardrail.v1 country pack.

Country-specific overrides
match the overlay's docstring:
  Straffeloven 311 child protection; Straffeloven 131 terrorism.
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "no"


def test_run_all_structural_assertions(no_overlay, no_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        no_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=no_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(no_overlay):
    override = A.get_override(no_overlay, 1)
    assert override["severity_floor"] == 5, (
        "NO: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_4(no_overlay):
    override = A.get_override(no_overlay, 4)
    assert override["severity_floor"] == 4, (
        "NO: category 4 override must be severity_floor 4"
    )


def test_legal_age_marketplace_alcohol_is_18(no_overlay):
    assert no_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 18


def test_legal_age_marketplace_tobacco_is_18(no_overlay):
    assert no_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 18


def test_protected_classes_includes_statutory_enumeration(no_overlay):
    classes = set(no_overlay["local_definitions"]["protected_classes"])
    expected = {'ethnic_origin', 'sex', 'religion', 'race'}
    assert expected <= classes, (
        "NO protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(no_overlay):
    langs = list(no_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['no', 'nb'], (
        f"NO primary_languages must equal ['no', 'nb']; got {langs}"
    )


def test_user_notice_opt_out_allowed_true(no_overlay):
    assert no_overlay["user_notice"]["opt_out_allowed"] is True


def test_election_rules_reference_authority(no_overlay):
    rules = no_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "no_valgdirektoratet_authority_v1"


def test_pack_passes_anti_misuse_validation(no_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(no_overlay)
    assert report.passed, f"NO pack failed anti-misuse: {report.errors}"
