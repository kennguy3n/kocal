"""Validate the kchat.jurisdiction.sa.guardrail.v1 country pack.

Country-specific overrides match
the overlay's docstring:
  Child Protection System (2014); Anti-Terrorism Law 2017; Anti-Cyber Crime Law 2007.
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "sa"


def test_run_all_structural_assertions(sa_overlay, sa_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        sa_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=sa_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(sa_overlay):
    override = A.get_override(sa_overlay, 1)
    assert override["severity_floor"] == 5, (
        "SA: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_5(sa_overlay):
    override = A.get_override(sa_overlay, 4)
    assert override["severity_floor"] == 5, (
        "SA: category 4 override must be severity_floor 5"
    )


def test_legal_age_marketplace_alcohol_is_21(sa_overlay):
    assert sa_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 21


def test_legal_age_marketplace_tobacco_is_18(sa_overlay):
    assert sa_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 18


def test_protected_classes_includes_statutory_enumeration(sa_overlay):
    classes = set(sa_overlay["local_definitions"]["protected_classes"])
    expected = {'religion', 'sex', 'disability', 'nationality'}
    assert expected <= classes, (
        "SA protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(sa_overlay):
    langs = list(sa_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['ar'], (
        f"SA primary_languages must equal ['ar']; got {langs}"
    )


def test_normalization_includes_arabic(sa_overlay):
    refs = list(
        sa_overlay["local_language_assets"]["normalization"]["transliteration_refs"]
    )
    assert "translit_arabic_v1" in refs, (
        f"SA normalization must include translit_arabic_v1; got {refs}"
    )


def test_user_notice_opt_out_allowed_false(sa_overlay):
    assert sa_overlay["user_notice"]["opt_out_allowed"] is False


def test_election_rules_reference_authority(sa_overlay):
    rules = sa_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "sa_gov_resource_authority_v1"


def test_pack_passes_anti_misuse_validation(sa_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(sa_overlay)
    assert report.passed, f"SA pack failed anti-misuse: {report.errors}"
