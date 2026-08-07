"""Validate the kchat.jurisdiction.co.guardrail.v1 country pack.

Country-specific overrides match
the overlay's docstring:
  Código de la Infancia y la Adolescencia (Ley 1098); anti-terrorism provisions.
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "co"


def test_run_all_structural_assertions(co_overlay, co_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        co_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=co_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(co_overlay):
    override = A.get_override(co_overlay, 1)
    assert override["severity_floor"] == 5, (
        "CO: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_4(co_overlay):
    override = A.get_override(co_overlay, 4)
    assert override["severity_floor"] == 4, (
        "CO: category 4 override must be severity_floor 4"
    )


def test_legal_age_marketplace_alcohol_is_18(co_overlay):
    assert co_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 18


def test_legal_age_marketplace_tobacco_is_18(co_overlay):
    assert co_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 18


def test_protected_classes_includes_statutory_enumeration(co_overlay):
    classes = set(co_overlay["local_definitions"]["protected_classes"])
    expected = {'sex', 'race', 'ethnic_origin', 'religion', 'disability'}
    assert expected <= classes, (
        "CO protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(co_overlay):
    langs = list(co_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['es'], (
        f"CO primary_languages must equal ['es']; got {langs}"
    )


def test_user_notice_opt_out_allowed_true(co_overlay):
    assert co_overlay["user_notice"]["opt_out_allowed"] is True


def test_election_rules_reference_authority(co_overlay):
    rules = co_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "co_registraduria_authority_v1"


def test_pack_passes_anti_misuse_validation(co_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(co_overlay)
    assert report.passed, f"CO pack failed anti-misuse: {report.errors}"
