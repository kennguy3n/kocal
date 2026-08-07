"""Validate the kchat.jurisdiction.uy.guardrail.v1 country pack.

Country-specific overrides
match the overlay's docstring:
  Codigo de la Ninez y la Adolescencia (CNA); Ley 19.293 anti-terrorism.
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "uy"


def test_run_all_structural_assertions(uy_overlay, uy_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        uy_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=uy_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(uy_overlay):
    override = A.get_override(uy_overlay, 1)
    assert override["severity_floor"] == 5, (
        "UY: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_4(uy_overlay):
    override = A.get_override(uy_overlay, 4)
    assert override["severity_floor"] == 4, (
        "UY: category 4 override must be severity_floor 4"
    )


def test_legal_age_marketplace_alcohol_is_18(uy_overlay):
    assert uy_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 18


def test_legal_age_marketplace_tobacco_is_18(uy_overlay):
    assert uy_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 18


def test_protected_classes_includes_statutory_enumeration(uy_overlay):
    classes = set(uy_overlay["local_definitions"]["protected_classes"])
    expected = {'ethnic_origin', 'sex', 'religion', 'race'}
    assert expected <= classes, (
        "UY protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(uy_overlay):
    langs = list(uy_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['es'], (
        f"UY primary_languages must equal ['es']; got {langs}"
    )


def test_user_notice_opt_out_allowed_true(uy_overlay):
    assert uy_overlay["user_notice"]["opt_out_allowed"] is True


def test_election_rules_reference_authority(uy_overlay):
    rules = uy_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "uy_corte_electoral_authority_v1"


def test_pack_passes_anti_misuse_validation(uy_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(uy_overlay)
    assert report.passed, f"UY pack failed anti-misuse: {report.errors}"
