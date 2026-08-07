"""Validate the kchat.jurisdiction.bd.guardrail.v1 country pack.

Country-specific overrides match
the overlay's docstring:
  Children Act 2013; Anti-Terrorism Act 2009; Digital Security Act.
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "bd"


def test_run_all_structural_assertions(bd_overlay, bd_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        bd_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=bd_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(bd_overlay):
    override = A.get_override(bd_overlay, 1)
    assert override["severity_floor"] == 5, (
        "BD: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_4(bd_overlay):
    override = A.get_override(bd_overlay, 4)
    assert override["severity_floor"] == 4, (
        "BD: category 4 override must be severity_floor 4"
    )


def test_legal_age_marketplace_alcohol_is_21(bd_overlay):
    assert bd_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 21


def test_legal_age_marketplace_tobacco_is_18(bd_overlay):
    assert bd_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 18


def test_protected_classes_includes_statutory_enumeration(bd_overlay):
    classes = set(bd_overlay["local_definitions"]["protected_classes"])
    expected = {'religion', 'sex', 'disability', 'race'}
    assert expected <= classes, (
        "BD protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(bd_overlay):
    langs = list(bd_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['bn'], (
        f"BD primary_languages must equal ['bn']; got {langs}"
    )


def test_normalization_includes_bengali(bd_overlay):
    refs = list(
        bd_overlay["local_language_assets"]["normalization"]["transliteration_refs"]
    )
    assert "translit_bengali_v1" in refs, (
        f"BD normalization must include translit_bengali_v1; got {refs}"
    )


def test_user_notice_opt_out_allowed_false(bd_overlay):
    assert bd_overlay["user_notice"]["opt_out_allowed"] is False


def test_election_rules_reference_authority(bd_overlay):
    rules = bd_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "bd_ec_authority_v1"


def test_pack_passes_anti_misuse_validation(bd_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(bd_overlay)
    assert report.passed, f"BD pack failed anti-misuse: {report.errors}"
