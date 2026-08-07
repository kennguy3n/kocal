"""Validate the kchat.jurisdiction.jp.guardrail.v1 country pack.

Country-specific overrides match
the overlay's docstring (Cannabis / Stimulants / Narcotics control
acts, Tokutei Shouhin Torihiki Hou, child-protection statute).
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "jp"


def test_run_all_structural_assertions(jp_overlay, jp_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        jp_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=jp_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(jp_overlay):
    override = A.get_override(jp_overlay, 1)
    assert override["severity_floor"] == 5, (
        "JP: CHILD_SAFETY override must pin severity_floor at 5"
    )


def test_category_11_drugs_weapons_floor_5(jp_overlay):
    override = A.get_override(jp_overlay, 11)
    assert override["severity_floor"] == 5, (
        "JP: DRUGS_WEAPONS override must be severity_floor 5 "
        "(Cannabis / Stimulants / Narcotics control + Firearm and Sword Law)"
    )


def test_category_7_scam_fraud_floor_4(jp_overlay):
    override = A.get_override(jp_overlay, 7)
    assert override["severity_floor"] == 4, (
        "JP: SCAM_FRAUD override must be severity_floor 4 (tokushoho)"
    )


def test_legal_age_marketplace_alcohol_is_20(jp_overlay):
    assert jp_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 20


def test_legal_age_marketplace_tobacco_is_20(jp_overlay):
    assert jp_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 20


def test_primary_languages_japanese(jp_overlay):
    langs = list(jp_overlay["local_language_assets"]["primary_languages"])
    assert langs == ["ja"], f"JP primary_languages must equal ['ja']; got {langs}"


def test_normalization_includes_translit_ja(jp_overlay):
    refs = list(
        jp_overlay["local_language_assets"]["normalization"]["transliteration_refs"]
    )
    assert "translit_ja_v1" in refs, (
        f"JP normalization must include translit_ja_v1; got {refs}"
    )


def test_protected_classes_article_14(jp_overlay):
    classes = set(jp_overlay["local_definitions"]["protected_classes"])
    expected = {"race", "creed", "sex", "social_status", "family_origin", "disability"}
    assert expected <= classes, (
        f"JP protected_classes must include Article 14 enumeration; "
        f"missing: {expected - classes}"
    )


def test_pack_passes_anti_misuse_validation(jp_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(jp_overlay)
    assert report.passed, f"JP pack failed anti-misuse: {report.errors}"
