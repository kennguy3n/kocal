"""Validate the kchat.jurisdiction.tw.guardrail.v1 country pack.

Country-specific overrides match
the overlay's docstring:
  Child and Youth Welfare and Protection Act; Anti-Infiltration Act / terrorism provisions.
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "tw"


def test_run_all_structural_assertions(tw_overlay, tw_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        tw_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=tw_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(tw_overlay):
    override = A.get_override(tw_overlay, 1)
    assert override["severity_floor"] == 5, (
        "TW: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_4(tw_overlay):
    override = A.get_override(tw_overlay, 4)
    assert override["severity_floor"] == 4, (
        "TW: category 4 override must be severity_floor 4"
    )


def test_legal_age_marketplace_alcohol_is_18(tw_overlay):
    assert tw_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 18


def test_legal_age_marketplace_tobacco_is_20(tw_overlay):
    assert tw_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 20


def test_protected_classes_includes_statutory_enumeration(tw_overlay):
    classes = set(tw_overlay["local_definitions"]["protected_classes"])
    expected = {'religion', 'sex', 'disability', 'race'}
    assert expected <= classes, (
        "TW protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(tw_overlay):
    langs = list(tw_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['zh'], (
        f"TW primary_languages must equal ['zh']; got {langs}"
    )


def test_normalization_includes_cjk(tw_overlay):
    refs = list(
        tw_overlay["local_language_assets"]["normalization"]["transliteration_refs"]
    )
    assert "translit_cjk_v1" in refs, (
        f"TW normalization must include translit_cjk_v1; got {refs}"
    )


def test_user_notice_opt_out_allowed_true(tw_overlay):
    assert tw_overlay["user_notice"]["opt_out_allowed"] is True


def test_election_rules_reference_authority(tw_overlay):
    rules = tw_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "tw_cec_authority_v1"


def test_pack_passes_anti_misuse_validation(tw_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(tw_overlay)
    assert report.passed, f"TW pack failed anti-misuse: {report.errors}"
