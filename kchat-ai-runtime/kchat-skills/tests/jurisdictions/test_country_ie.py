"""Validate the kchat.jurisdiction.ie.guardrail.v1 country pack.

Country-specific overrides
match the overlay's docstring:
  Online Safety and Media Regulation Act 2022; Child Trafficking and Pornography Act 1998; Offences against the State Act 1939.
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "ie"


def test_run_all_structural_assertions(ie_overlay, ie_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        ie_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=ie_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(ie_overlay):
    override = A.get_override(ie_overlay, 1)
    assert override["severity_floor"] == 5, (
        "IE: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_4(ie_overlay):
    override = A.get_override(ie_overlay, 4)
    assert override["severity_floor"] == 4, (
        "IE: category 4 override must be severity_floor 4"
    )


def test_legal_age_marketplace_alcohol_is_18(ie_overlay):
    assert ie_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 18


def test_legal_age_marketplace_tobacco_is_18(ie_overlay):
    assert ie_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 18


def test_protected_classes_includes_statutory_enumeration(ie_overlay):
    classes = set(ie_overlay["local_definitions"]["protected_classes"])
    expected = {'sex', 'ethnic_origin', 'disability', 'religion', 'race'}
    assert expected <= classes, (
        "IE protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(ie_overlay):
    langs = list(ie_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['en', 'ga'], (
        f"IE primary_languages must equal ['en', 'ga']; got {langs}"
    )


def test_user_notice_opt_out_allowed_true(ie_overlay):
    assert ie_overlay["user_notice"]["opt_out_allowed"] is True


def test_election_rules_reference_authority(ie_overlay):
    rules = ie_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "ie_coimisiun_na_mean_authority_v1"


def test_pack_passes_anti_misuse_validation(ie_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(ie_overlay)
    assert report.passed, f"IE pack failed anti-misuse: {report.errors}"
