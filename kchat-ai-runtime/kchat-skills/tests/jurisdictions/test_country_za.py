"""Validate the kchat.jurisdiction.za.guardrail.v1 country pack.

Country-specific overrides match
the overlay's docstring:
  Children's Act 38 of 2005; POCDATARA 2004; Films and Publications Act.
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "za"


def test_run_all_structural_assertions(za_overlay, za_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        za_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=za_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(za_overlay):
    override = A.get_override(za_overlay, 1)
    assert override["severity_floor"] == 5, (
        "ZA: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_4(za_overlay):
    override = A.get_override(za_overlay, 4)
    assert override["severity_floor"] == 4, (
        "ZA: category 4 override must be severity_floor 4"
    )


def test_legal_age_marketplace_alcohol_is_18(za_overlay):
    assert za_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 18


def test_legal_age_marketplace_tobacco_is_18(za_overlay):
    assert za_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 18


def test_protected_classes_includes_statutory_enumeration(za_overlay):
    classes = set(za_overlay["local_definitions"]["protected_classes"])
    expected = {'sex', 'sexual_orientation', 'race', 'religion', 'disability'}
    assert expected <= classes, (
        "ZA protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(za_overlay):
    langs = list(za_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['en', 'af', 'zu'], (
        f"ZA primary_languages must equal ['en', 'af', 'zu']; got {langs}"
    )


def test_user_notice_opt_out_allowed_true(za_overlay):
    assert za_overlay["user_notice"]["opt_out_allowed"] is True


def test_election_rules_reference_authority(za_overlay):
    rules = za_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "za_iec_authority_v1"


def test_pack_passes_anti_misuse_validation(za_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(za_overlay)
    assert report.passed, f"ZA pack failed anti-misuse: {report.errors}"
