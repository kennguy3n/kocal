"""Validate the kchat.jurisdiction.ae.guardrail.v1 country pack.

Country-specific overrides match
the overlay's docstring:
  Wadeema's Law (Federal Law 3/2016); Federal Decree-Law 7/2014 anti-terrorism; Cybercrimes Law.
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "ae"


def test_run_all_structural_assertions(ae_overlay, ae_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        ae_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=ae_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(ae_overlay):
    override = A.get_override(ae_overlay, 1)
    assert override["severity_floor"] == 5, (
        "AE: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_5(ae_overlay):
    override = A.get_override(ae_overlay, 4)
    assert override["severity_floor"] == 5, (
        "AE: category 4 override must be severity_floor 5"
    )


def test_legal_age_marketplace_alcohol_is_21(ae_overlay):
    assert ae_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 21


def test_legal_age_marketplace_tobacco_is_18(ae_overlay):
    assert ae_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 18


def test_protected_classes_includes_statutory_enumeration(ae_overlay):
    classes = set(ae_overlay["local_definitions"]["protected_classes"])
    expected = {'religion', 'sex', 'race', 'nationality'}
    assert expected <= classes, (
        "AE protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(ae_overlay):
    langs = list(ae_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['ar', 'en'], (
        f"AE primary_languages must equal ['ar', 'en']; got {langs}"
    )


def test_normalization_includes_arabic(ae_overlay):
    refs = list(
        ae_overlay["local_language_assets"]["normalization"]["transliteration_refs"]
    )
    assert "translit_arabic_v1" in refs, (
        f"AE normalization must include translit_arabic_v1; got {refs}"
    )


def test_user_notice_opt_out_allowed_false(ae_overlay):
    assert ae_overlay["user_notice"]["opt_out_allowed"] is False


def test_election_rules_reference_authority(ae_overlay):
    rules = ae_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "ae_nec_authority_v1"


def test_pack_passes_anti_misuse_validation(ae_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(ae_overlay)
    assert report.passed, f"AE pack failed anti-misuse: {report.errors}"
