"""Validate the kchat.jurisdiction.ch.guardrail.v1 country pack.

Country-specific overrides match
the overlay's docstring:
  StGB child protection; StGB Art. 261bis anti-racism.
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "ch"


def test_run_all_structural_assertions(ch_overlay, ch_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        ch_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=ch_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(ch_overlay):
    override = A.get_override(ch_overlay, 1)
    assert override["severity_floor"] == 5, (
        "CH: category 1 override must be severity_floor 5"
    )


def test_category_6_hate_floor_4(ch_overlay):
    override = A.get_override(ch_overlay, 6)
    assert override["severity_floor"] == 4, (
        "CH: category 6 override must be severity_floor 4"
    )


def test_legal_age_marketplace_alcohol_is_16(ch_overlay):
    assert ch_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 16


def test_legal_age_marketplace_tobacco_is_18(ch_overlay):
    assert ch_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 18


def test_protected_classes_includes_statutory_enumeration(ch_overlay):
    classes = set(ch_overlay["local_definitions"]["protected_classes"])
    expected = {'religion', 'sexual_orientation', 'race', 'ethnic_origin'}
    assert expected <= classes, (
        "CH protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(ch_overlay):
    langs = list(ch_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['de', 'fr', 'it', 'rm'], (
        f"CH primary_languages must equal ['de', 'fr', 'it', 'rm']; got {langs}"
    )


def test_user_notice_opt_out_allowed_true(ch_overlay):
    assert ch_overlay["user_notice"]["opt_out_allowed"] is True


def test_election_rules_reference_authority(ch_overlay):
    rules = ch_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "ch_bundeskanzlei_authority_v1"


def test_pack_passes_anti_misuse_validation(ch_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(ch_overlay)
    assert report.passed, f"CH pack failed anti-misuse: {report.errors}"
