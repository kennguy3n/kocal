"""Validate the kchat.jurisdiction.pe.guardrail.v1 country pack.

Country-specific overrides match
the overlay's docstring:
  Código de los Niños y Adolescentes (Ley 27.337).
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "pe"


def test_run_all_structural_assertions(pe_overlay, pe_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        pe_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=pe_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(pe_overlay):
    override = A.get_override(pe_overlay, 1)
    assert override["severity_floor"] == 5, (
        "PE: category 1 override must be severity_floor 5"
    )


def test_legal_age_marketplace_alcohol_is_18(pe_overlay):
    assert pe_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 18


def test_legal_age_marketplace_tobacco_is_18(pe_overlay):
    assert pe_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 18


def test_protected_classes_includes_statutory_enumeration(pe_overlay):
    classes = set(pe_overlay["local_definitions"]["protected_classes"])
    expected = {'sex', 'race', 'ethnic_origin', 'religion', 'disability'}
    assert expected <= classes, (
        "PE protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(pe_overlay):
    langs = list(pe_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['es'], (
        f"PE primary_languages must equal ['es']; got {langs}"
    )


def test_user_notice_opt_out_allowed_true(pe_overlay):
    assert pe_overlay["user_notice"]["opt_out_allowed"] is True


def test_election_rules_reference_authority(pe_overlay):
    rules = pe_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "pe_onpe_authority_v1"


def test_pack_passes_anti_misuse_validation(pe_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(pe_overlay)
    assert report.passed, f"PE pack failed anti-misuse: {report.errors}"
