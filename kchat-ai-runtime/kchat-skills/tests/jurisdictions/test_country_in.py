"""Validate the kchat.jurisdiction.in.guardrail.v1 country pack.

Country-specific overrides match
the overlay's docstring (UAPA, IPC §153A / §295A, IT Act §67 / §67A).
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "in"


def test_run_all_structural_assertions(in_overlay, in_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        in_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=in_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_4_extremism_floor_4(in_overlay):
    override = A.get_override(in_overlay, 4)
    assert override["severity_floor"] == 4, "IN: EXTREMISM (UAPA) must be 4"


def test_category_6_hate_floor_4(in_overlay):
    override = A.get_override(in_overlay, 6)
    assert override["severity_floor"] == 4, (
        "IN: HATE (IPC §153A / §295A) must be 4"
    )


def test_category_10_sexual_adult_floor_5(in_overlay):
    override = A.get_override(in_overlay, 10)
    assert override["severity_floor"] == 5, (
        "IN: SEXUAL_ADULT (IT Act §67 / §67A) must be 5"
    )


def test_primary_languages_hindi_and_indian_english(in_overlay):
    langs = list(in_overlay["local_language_assets"]["primary_languages"])
    assert "hi" in langs and "en-IN" in langs, (
        f"IN primary_languages must include hi and en-IN; got {langs}"
    )


def test_normalization_includes_devanagari_translit(in_overlay):
    refs = list(
        in_overlay["local_language_assets"]["normalization"]["transliteration_refs"]
    )
    assert "translit_devanagari_v1" in refs, (
        f"IN normalization must include translit_devanagari_v1; got {refs}"
    )


def test_protected_classes_articles_15_16(in_overlay):
    classes = set(in_overlay["local_definitions"]["protected_classes"])
    expected = {"race", "religion", "caste", "sex", "place_of_birth", "disability"}
    assert expected <= classes, (
        f"IN protected_classes must include Articles 15-16 enumeration; "
        f"missing: {expected - classes}"
    )


def test_alcohol_age_conservative_default_21(in_overlay):
    assert in_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 21


def test_election_rules_reference_eci(in_overlay):
    rules = in_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "in_eci_authority_v1"


def test_pack_passes_anti_misuse_validation(in_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(in_overlay)
    assert report.passed, f"IN pack failed anti-misuse: {report.errors}"
