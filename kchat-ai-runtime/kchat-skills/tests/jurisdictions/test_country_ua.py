"""Validate the kchat.jurisdiction.ua.guardrail.v1 country pack.

Country-specific overrides
match the overlay's docstring:
  Law on Child Protection (1995); Law on Combating Terrorism (2003).
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "ua"


def test_run_all_structural_assertions(ua_overlay, ua_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        ua_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=ua_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(ua_overlay):
    override = A.get_override(ua_overlay, 1)
    assert override["severity_floor"] == 5, (
        "UA: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_4(ua_overlay):
    override = A.get_override(ua_overlay, 4)
    assert override["severity_floor"] == 4, (
        "UA: category 4 override must be severity_floor 4"
    )


def test_legal_age_marketplace_alcohol_is_18(ua_overlay):
    assert ua_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 18


def test_legal_age_marketplace_tobacco_is_18(ua_overlay):
    assert ua_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 18


def test_protected_classes_includes_statutory_enumeration(ua_overlay):
    classes = set(ua_overlay["local_definitions"]["protected_classes"])
    expected = {'ethnic_origin', 'sex', 'religion', 'race'}
    assert expected <= classes, (
        "UA protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(ua_overlay):
    langs = list(ua_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['uk', 'ru'], (
        f"UA primary_languages must equal ['uk', 'ru']; got {langs}"
    )


def test_user_notice_opt_out_allowed_true(ua_overlay):
    assert ua_overlay["user_notice"]["opt_out_allowed"] is True


def test_election_rules_reference_authority(ua_overlay):
    rules = ua_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "ua_cvk_authority_v1"


def test_pack_passes_anti_misuse_validation(ua_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(ua_overlay)
    assert report.passed, f"UA pack failed anti-misuse: {report.errors}"
