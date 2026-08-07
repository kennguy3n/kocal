"""Validate the kchat.jurisdiction.id.guardrail.v1 country pack.

Country-specific overrides match
the overlay's docstring:
  UU ITE; UU 35/2014 child protection; Anti-Terrorism Law; UU 44/2008 pornography law.
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "id"


def test_run_all_structural_assertions(id_overlay, id_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        id_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=id_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(id_overlay):
    override = A.get_override(id_overlay, 1)
    assert override["severity_floor"] == 5, (
        "ID: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_4(id_overlay):
    override = A.get_override(id_overlay, 4)
    assert override["severity_floor"] == 4, (
        "ID: category 4 override must be severity_floor 4"
    )


def test_category_10_sexual_adult_floor_5(id_overlay):
    override = A.get_override(id_overlay, 10)
    assert override["severity_floor"] == 5, (
        "ID: category 10 override must be severity_floor 5"
    )


def test_legal_age_marketplace_alcohol_is_21(id_overlay):
    assert id_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 21


def test_legal_age_marketplace_tobacco_is_18(id_overlay):
    assert id_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 18


def test_protected_classes_includes_statutory_enumeration(id_overlay):
    classes = set(id_overlay["local_definitions"]["protected_classes"])
    expected = {'sex', 'race', 'ethnic_origin', 'religion', 'disability'}
    assert expected <= classes, (
        "ID protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(id_overlay):
    langs = list(id_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['id'], (
        f"ID primary_languages must equal ['id']; got {langs}"
    )


def test_user_notice_opt_out_allowed_true(id_overlay):
    assert id_overlay["user_notice"]["opt_out_allowed"] is True


def test_election_rules_reference_authority(id_overlay):
    rules = id_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "id_kpu_authority_v1"


def test_pack_passes_anti_misuse_validation(id_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(id_overlay)
    assert report.passed, f"ID pack failed anti-misuse: {report.errors}"
