"""Validate the kchat.jurisdiction.sg.guardrail.v1 country pack.

Country-specific overrides match
the overlay's docstring:
  Children and Young Persons Act; Internal Security Act; Online Safety Acts.
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "sg"


def test_run_all_structural_assertions(sg_overlay, sg_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        sg_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=sg_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(sg_overlay):
    override = A.get_override(sg_overlay, 1)
    assert override["severity_floor"] == 5, (
        "SG: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_4(sg_overlay):
    override = A.get_override(sg_overlay, 4)
    assert override["severity_floor"] == 4, (
        "SG: category 4 override must be severity_floor 4"
    )


def test_legal_age_marketplace_alcohol_is_18(sg_overlay):
    assert sg_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 18


def test_legal_age_marketplace_tobacco_is_21(sg_overlay):
    assert sg_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 21


def test_protected_classes_includes_statutory_enumeration(sg_overlay):
    classes = set(sg_overlay["local_definitions"]["protected_classes"])
    expected = {'religion', 'race', 'disability', 'language'}
    assert expected <= classes, (
        "SG protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(sg_overlay):
    langs = list(sg_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['en', 'zh', 'ms', 'ta'], (
        f"SG primary_languages must equal ['en', 'zh', 'ms', 'ta']; got {langs}"
    )


def test_normalization_includes_cjk(sg_overlay):
    refs = list(
        sg_overlay["local_language_assets"]["normalization"]["transliteration_refs"]
    )
    assert "translit_cjk_v1" in refs, (
        f"SG normalization must include translit_cjk_v1; got {refs}"
    )


def test_user_notice_opt_out_allowed_false(sg_overlay):
    assert sg_overlay["user_notice"]["opt_out_allowed"] is False


def test_election_rules_reference_authority(sg_overlay):
    rules = sg_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "sg_eld_authority_v1"


def test_pack_passes_anti_misuse_validation(sg_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(sg_overlay)
    assert report.passed, f"SG pack failed anti-misuse: {report.errors}"
