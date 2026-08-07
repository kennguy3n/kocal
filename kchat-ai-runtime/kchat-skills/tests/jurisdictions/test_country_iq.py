"""Validate the kchat.jurisdiction.iq.guardrail.v1 country pack.

Country-specific overrides
match the overlay's docstring:
  Juvenile Welfare Law No. 76/1983; Anti-Terrorism Law No. 13/2005.
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "iq"


def test_run_all_structural_assertions(iq_overlay, iq_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        iq_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=iq_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(iq_overlay):
    override = A.get_override(iq_overlay, 1)
    assert override["severity_floor"] == 5, (
        "IQ: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_4(iq_overlay):
    override = A.get_override(iq_overlay, 4)
    assert override["severity_floor"] == 4, (
        "IQ: category 4 override must be severity_floor 4"
    )


def test_legal_age_marketplace_alcohol_is_18(iq_overlay):
    assert iq_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 18


def test_legal_age_marketplace_tobacco_is_18(iq_overlay):
    assert iq_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 18


def test_protected_classes_includes_statutory_enumeration(iq_overlay):
    classes = set(iq_overlay["local_definitions"]["protected_classes"])
    expected = {'ethnic_origin', 'sex', 'religion', 'race'}
    assert expected <= classes, (
        "IQ protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(iq_overlay):
    langs = list(iq_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['ar', 'ku'], (
        f"IQ primary_languages must equal ['ar', 'ku']; got {langs}"
    )


def test_user_notice_opt_out_allowed_true(iq_overlay):
    assert iq_overlay["user_notice"]["opt_out_allowed"] is True


def test_election_rules_reference_authority(iq_overlay):
    rules = iq_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "iq_ihec_authority_v1"


def test_pack_passes_anti_misuse_validation(iq_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(iq_overlay)
    assert report.passed, f"IQ pack failed anti-misuse: {report.errors}"
