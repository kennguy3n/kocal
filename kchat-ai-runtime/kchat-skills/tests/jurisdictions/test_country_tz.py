"""Validate the kchat.jurisdiction.tz.guardrail.v1 country pack.

Country-specific overrides
match the overlay's docstring:
  Law of the Child Act 2009; Prevention of Terrorism Act 2002.
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "tz"


def test_run_all_structural_assertions(tz_overlay, tz_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        tz_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=tz_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(tz_overlay):
    override = A.get_override(tz_overlay, 1)
    assert override["severity_floor"] == 5, (
        "TZ: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_4(tz_overlay):
    override = A.get_override(tz_overlay, 4)
    assert override["severity_floor"] == 4, (
        "TZ: category 4 override must be severity_floor 4"
    )


def test_legal_age_marketplace_alcohol_is_18(tz_overlay):
    assert tz_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 18


def test_legal_age_marketplace_tobacco_is_18(tz_overlay):
    assert tz_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 18


def test_protected_classes_includes_statutory_enumeration(tz_overlay):
    classes = set(tz_overlay["local_definitions"]["protected_classes"])
    expected = {'ethnic_origin', 'sex', 'religion', 'race'}
    assert expected <= classes, (
        "TZ protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(tz_overlay):
    langs = list(tz_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['sw', 'en'], (
        f"TZ primary_languages must equal ['sw', 'en']; got {langs}"
    )


def test_user_notice_opt_out_allowed_true(tz_overlay):
    assert tz_overlay["user_notice"]["opt_out_allowed"] is True


def test_election_rules_reference_authority(tz_overlay):
    rules = tz_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "tz_nec_authority_v1"


def test_pack_passes_anti_misuse_validation(tz_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(tz_overlay)
    assert report.passed, f"TZ pack failed anti-misuse: {report.errors}"
