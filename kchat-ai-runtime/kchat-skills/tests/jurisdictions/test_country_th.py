"""Validate the kchat.jurisdiction.th.guardrail.v1 country pack.

Country-specific overrides match
the overlay's docstring:
  Child Protection Act B.E. 2546; Computer Crime Act; lèse-majesté (Criminal Code §112).
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "th"


def test_run_all_structural_assertions(th_overlay, th_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        th_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=th_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(th_overlay):
    override = A.get_override(th_overlay, 1)
    assert override["severity_floor"] == 5, (
        "TH: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_4(th_overlay):
    override = A.get_override(th_overlay, 4)
    assert override["severity_floor"] == 4, (
        "TH: category 4 override must be severity_floor 4"
    )


def test_category_6_hate_floor_5(th_overlay):
    override = A.get_override(th_overlay, 6)
    assert override["severity_floor"] == 5, (
        "TH: category 6 override must be severity_floor 5"
    )


def test_legal_age_marketplace_alcohol_is_20(th_overlay):
    assert th_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 20


def test_legal_age_marketplace_tobacco_is_20(th_overlay):
    assert th_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 20


def test_protected_classes_includes_statutory_enumeration(th_overlay):
    classes = set(th_overlay["local_definitions"]["protected_classes"])
    expected = {'religious_belief', 'sex', 'disability', 'race'}
    assert expected <= classes, (
        "TH protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(th_overlay):
    langs = list(th_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['th'], (
        f"TH primary_languages must equal ['th']; got {langs}"
    )


def test_normalization_includes_thai(th_overlay):
    refs = list(
        th_overlay["local_language_assets"]["normalization"]["transliteration_refs"]
    )
    assert "translit_thai_v1" in refs, (
        f"TH normalization must include translit_thai_v1; got {refs}"
    )


def test_user_notice_opt_out_allowed_false(th_overlay):
    assert th_overlay["user_notice"]["opt_out_allowed"] is False


def test_election_rules_reference_authority(th_overlay):
    rules = th_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "th_ect_authority_v1"


def test_pack_passes_anti_misuse_validation(th_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(th_overlay)
    assert report.passed, f"TH pack failed anti-misuse: {report.errors}"
