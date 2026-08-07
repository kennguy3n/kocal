"""Validate the kchat.jurisdiction.ph.guardrail.v1 country pack.

Country-specific overrides match
the overlay's docstring:
  RA 7610 child protection; Human Security Act / RA 11479 anti-terrorism.
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "ph"


def test_run_all_structural_assertions(ph_overlay, ph_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        ph_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=ph_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(ph_overlay):
    override = A.get_override(ph_overlay, 1)
    assert override["severity_floor"] == 5, (
        "PH: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_4(ph_overlay):
    override = A.get_override(ph_overlay, 4)
    assert override["severity_floor"] == 4, (
        "PH: category 4 override must be severity_floor 4"
    )


def test_legal_age_marketplace_alcohol_is_18(ph_overlay):
    assert ph_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 18


def test_legal_age_marketplace_tobacco_is_21(ph_overlay):
    assert ph_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 21


def test_protected_classes_includes_statutory_enumeration(ph_overlay):
    classes = set(ph_overlay["local_definitions"]["protected_classes"])
    expected = {'religion', 'sex', 'disability', 'race'}
    assert expected <= classes, (
        "PH protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(ph_overlay):
    langs = list(ph_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['en', 'tl'], (
        f"PH primary_languages must equal ['en', 'tl']; got {langs}"
    )


def test_user_notice_opt_out_allowed_true(ph_overlay):
    assert ph_overlay["user_notice"]["opt_out_allowed"] is True


def test_election_rules_reference_authority(ph_overlay):
    rules = ph_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "ph_comelec_authority_v1"


def test_pack_passes_anti_misuse_validation(ph_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(ph_overlay)
    assert report.passed, f"PH pack failed anti-misuse: {report.errors}"
