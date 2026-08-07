"""Validate the kchat.jurisdiction.pk.guardrail.v1 country pack.

Country-specific overrides match
the overlay's docstring:
  Pakistan Penal Code child protection; Anti-Terrorism Act 1997; PECA 2016.
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "pk"


def test_run_all_structural_assertions(pk_overlay, pk_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        pk_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=pk_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(pk_overlay):
    override = A.get_override(pk_overlay, 1)
    assert override["severity_floor"] == 5, (
        "PK: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_4(pk_overlay):
    override = A.get_override(pk_overlay, 4)
    assert override["severity_floor"] == 4, (
        "PK: category 4 override must be severity_floor 4"
    )


def test_legal_age_marketplace_alcohol_is_21(pk_overlay):
    assert pk_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 21


def test_legal_age_marketplace_tobacco_is_18(pk_overlay):
    assert pk_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 18


def test_protected_classes_includes_statutory_enumeration(pk_overlay):
    classes = set(pk_overlay["local_definitions"]["protected_classes"])
    expected = {'religion', 'sex', 'disability', 'race'}
    assert expected <= classes, (
        "PK protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(pk_overlay):
    langs = list(pk_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['ur', 'en'], (
        f"PK primary_languages must equal ['ur', 'en']; got {langs}"
    )


def test_normalization_includes_arabic(pk_overlay):
    refs = list(
        pk_overlay["local_language_assets"]["normalization"]["transliteration_refs"]
    )
    assert "translit_arabic_v1" in refs, (
        f"PK normalization must include translit_arabic_v1; got {refs}"
    )


def test_user_notice_opt_out_allowed_false(pk_overlay):
    assert pk_overlay["user_notice"]["opt_out_allowed"] is False


def test_election_rules_reference_authority(pk_overlay):
    rules = pk_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "pk_ecp_authority_v1"


def test_pack_passes_anti_misuse_validation(pk_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(pk_overlay)
    assert report.passed, f"PK pack failed anti-misuse: {report.errors}"
