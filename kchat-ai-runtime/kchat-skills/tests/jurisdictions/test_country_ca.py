"""Validate the kchat.jurisdiction.ca.guardrail.v1 country pack.

Country-specific overrides match
the overlay's docstring:
  Criminal Code child exploitation, Criminal Code terrorism, Competition Act fraud.
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "ca"


def test_run_all_structural_assertions(ca_overlay, ca_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        ca_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=ca_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(ca_overlay):
    override = A.get_override(ca_overlay, 1)
    assert override["severity_floor"] == 5, (
        "CA: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_4(ca_overlay):
    override = A.get_override(ca_overlay, 4)
    assert override["severity_floor"] == 4, (
        "CA: category 4 override must be severity_floor 4"
    )


def test_category_7_scam_fraud_floor_3(ca_overlay):
    override = A.get_override(ca_overlay, 7)
    assert override["severity_floor"] == 3, (
        "CA: category 7 override must be severity_floor 3"
    )


def test_legal_age_marketplace_alcohol_is_19(ca_overlay):
    assert ca_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 19


def test_legal_age_marketplace_tobacco_is_19(ca_overlay):
    assert ca_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 19


def test_protected_classes_includes_statutory_enumeration(ca_overlay):
    classes = set(ca_overlay["local_definitions"]["protected_classes"])
    expected = {'sex', 'race', 'religion', 'national_or_ethnic_origin', 'disability'}
    assert expected <= classes, (
        "CA protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(ca_overlay):
    langs = list(ca_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['en', 'fr'], (
        f"CA primary_languages must equal ['en', 'fr']; got {langs}"
    )


def test_user_notice_opt_out_allowed_true(ca_overlay):
    assert ca_overlay["user_notice"]["opt_out_allowed"] is True


def test_election_rules_reference_authority(ca_overlay):
    rules = ca_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "ca_elections_canada_authority_v1"


def test_pack_passes_anti_misuse_validation(ca_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(ca_overlay)
    assert report.passed, f"CA pack failed anti-misuse: {report.errors}"
