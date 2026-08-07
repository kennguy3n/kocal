"""Validate the kchat.jurisdiction.fr.guardrail.v1 country pack.

Country-specific overrides match
the overlay's docstring:
  Loi Avia / DSA transposition (apologie du terrorisme); loi Gayssot (hate); child protection.
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "fr"


def test_run_all_structural_assertions(fr_overlay, fr_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        fr_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=fr_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(fr_overlay):
    override = A.get_override(fr_overlay, 1)
    assert override["severity_floor"] == 5, (
        "FR: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_5(fr_overlay):
    override = A.get_override(fr_overlay, 4)
    assert override["severity_floor"] == 5, (
        "FR: category 4 override must be severity_floor 5"
    )


def test_category_6_hate_floor_4(fr_overlay):
    override = A.get_override(fr_overlay, 6)
    assert override["severity_floor"] == 4, (
        "FR: category 6 override must be severity_floor 4"
    )


def test_legal_age_marketplace_alcohol_is_18(fr_overlay):
    assert fr_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 18


def test_legal_age_marketplace_tobacco_is_18(fr_overlay):
    assert fr_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 18


def test_protected_classes_includes_statutory_enumeration(fr_overlay):
    classes = set(fr_overlay["local_definitions"]["protected_classes"])
    expected = {'sex', 'race', 'origin', 'religion', 'disability'}
    assert expected <= classes, (
        "FR protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(fr_overlay):
    langs = list(fr_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['fr'], (
        f"FR primary_languages must equal ['fr']; got {langs}"
    )


def test_user_notice_opt_out_allowed_true(fr_overlay):
    assert fr_overlay["user_notice"]["opt_out_allowed"] is True


def test_election_rules_reference_authority(fr_overlay):
    rules = fr_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "fr_ministere_interieur_authority_v1"


def test_pack_passes_anti_misuse_validation(fr_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(fr_overlay)
    assert report.passed, f"FR pack failed anti-misuse: {report.errors}"
