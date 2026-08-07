"""Validate the kchat.jurisdiction.cz.guardrail.v1 country pack.

Country-specific overrides
match the overlay's docstring:
  Trestni zakonik child protection (192-193); 311 anti-terrorism.
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "cz"


def test_run_all_structural_assertions(cz_overlay, cz_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        cz_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=cz_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(cz_overlay):
    override = A.get_override(cz_overlay, 1)
    assert override["severity_floor"] == 5, (
        "CZ: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_4(cz_overlay):
    override = A.get_override(cz_overlay, 4)
    assert override["severity_floor"] == 4, (
        "CZ: category 4 override must be severity_floor 4"
    )


def test_legal_age_marketplace_alcohol_is_18(cz_overlay):
    assert cz_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 18


def test_legal_age_marketplace_tobacco_is_18(cz_overlay):
    assert cz_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 18


def test_protected_classes_includes_statutory_enumeration(cz_overlay):
    classes = set(cz_overlay["local_definitions"]["protected_classes"])
    expected = {'ethnic_origin', 'sex', 'religion', 'race'}
    assert expected <= classes, (
        "CZ protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(cz_overlay):
    langs = list(cz_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['cs'], (
        f"CZ primary_languages must equal ['cs']; got {langs}"
    )


def test_user_notice_opt_out_allowed_true(cz_overlay):
    assert cz_overlay["user_notice"]["opt_out_allowed"] is True


def test_election_rules_reference_authority(cz_overlay):
    rules = cz_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "cz_csu_authority_v1"


def test_pack_passes_anti_misuse_validation(cz_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(cz_overlay)
    assert report.passed, f"CZ pack failed anti-misuse: {report.errors}"
