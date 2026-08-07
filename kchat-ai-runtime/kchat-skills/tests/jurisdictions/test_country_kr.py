"""Validate the kchat.jurisdiction.kr.guardrail.v1 country pack.

Country-specific overrides match
the overlay's docstring:
  Act on the Protection of Children and Youth; National Security Act; KCSC content regulation.
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "kr"


def test_run_all_structural_assertions(kr_overlay, kr_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        kr_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=kr_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(kr_overlay):
    override = A.get_override(kr_overlay, 1)
    assert override["severity_floor"] == 5, (
        "KR: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_4(kr_overlay):
    override = A.get_override(kr_overlay, 4)
    assert override["severity_floor"] == 4, (
        "KR: category 4 override must be severity_floor 4"
    )


def test_legal_age_marketplace_alcohol_is_19(kr_overlay):
    assert kr_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 19


def test_legal_age_marketplace_tobacco_is_19(kr_overlay):
    assert kr_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 19


def test_protected_classes_includes_statutory_enumeration(kr_overlay):
    classes = set(kr_overlay["local_definitions"]["protected_classes"])
    expected = {'sex', 'race', 'religion', 'disability', 'nationality'}
    assert expected <= classes, (
        "KR protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(kr_overlay):
    langs = list(kr_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['ko'], (
        f"KR primary_languages must equal ['ko']; got {langs}"
    )


def test_normalization_includes_hangul(kr_overlay):
    refs = list(
        kr_overlay["local_language_assets"]["normalization"]["transliteration_refs"]
    )
    assert "translit_hangul_v1" in refs, (
        f"KR normalization must include translit_hangul_v1; got {refs}"
    )


def test_user_notice_opt_out_allowed_true(kr_overlay):
    assert kr_overlay["user_notice"]["opt_out_allowed"] is True


def test_election_rules_reference_authority(kr_overlay):
    rules = kr_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "kr_nec_authority_v1"


def test_pack_passes_anti_misuse_validation(kr_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(kr_overlay)
    assert report.passed, f"KR pack failed anti-misuse: {report.errors}"
