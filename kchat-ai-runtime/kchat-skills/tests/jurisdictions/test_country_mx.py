"""Validate the kchat.jurisdiction.mx.guardrail.v1 country pack.

Country-specific overrides match
the overlay's docstring:
  LGPNNA child protection, Ley Federal contra la Delincuencia Organizada, COFEPRIS drug regulation.
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "mx"


def test_run_all_structural_assertions(mx_overlay, mx_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        mx_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=mx_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(mx_overlay):
    override = A.get_override(mx_overlay, 1)
    assert override["severity_floor"] == 5, (
        "MX: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_4(mx_overlay):
    override = A.get_override(mx_overlay, 4)
    assert override["severity_floor"] == 4, (
        "MX: category 4 override must be severity_floor 4"
    )


def test_category_11_drugs_weapons_floor_4(mx_overlay):
    override = A.get_override(mx_overlay, 11)
    assert override["severity_floor"] == 4, (
        "MX: category 11 override must be severity_floor 4"
    )


def test_legal_age_marketplace_alcohol_is_18(mx_overlay):
    assert mx_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 18


def test_legal_age_marketplace_tobacco_is_18(mx_overlay):
    assert mx_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 18


def test_protected_classes_includes_statutory_enumeration(mx_overlay):
    classes = set(mx_overlay["local_definitions"]["protected_classes"])
    expected = {'sex', 'ethnic_or_national_origin', 'disability', 'religion'}
    assert expected <= classes, (
        "MX protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(mx_overlay):
    langs = list(mx_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['es'], (
        f"MX primary_languages must equal ['es']; got {langs}"
    )


def test_user_notice_opt_out_allowed_true(mx_overlay):
    assert mx_overlay["user_notice"]["opt_out_allowed"] is True


def test_election_rules_reference_authority(mx_overlay):
    rules = mx_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "mx_ine_authority_v1"


def test_pack_passes_anti_misuse_validation(mx_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(mx_overlay)
    assert report.passed, f"MX pack failed anti-misuse: {report.errors}"
