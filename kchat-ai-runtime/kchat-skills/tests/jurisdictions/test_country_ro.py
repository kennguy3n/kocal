"""Validate the kchat.jurisdiction.ro.guardrail.v1 country pack.

Country-specific overrides
match the overlay's docstring:
  Legea 272/2004 child protection; Legea 535/2004 anti-terrorism.
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "ro"


def test_run_all_structural_assertions(ro_overlay, ro_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        ro_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=ro_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(ro_overlay):
    override = A.get_override(ro_overlay, 1)
    assert override["severity_floor"] == 5, (
        "RO: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_4(ro_overlay):
    override = A.get_override(ro_overlay, 4)
    assert override["severity_floor"] == 4, (
        "RO: category 4 override must be severity_floor 4"
    )


def test_legal_age_marketplace_alcohol_is_18(ro_overlay):
    assert ro_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 18


def test_legal_age_marketplace_tobacco_is_18(ro_overlay):
    assert ro_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 18


def test_protected_classes_includes_statutory_enumeration(ro_overlay):
    classes = set(ro_overlay["local_definitions"]["protected_classes"])
    expected = {'ethnic_origin', 'sex', 'religion', 'race'}
    assert expected <= classes, (
        "RO protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(ro_overlay):
    langs = list(ro_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['ro'], (
        f"RO primary_languages must equal ['ro']; got {langs}"
    )


def test_user_notice_opt_out_allowed_true(ro_overlay):
    assert ro_overlay["user_notice"]["opt_out_allowed"] is True


def test_election_rules_reference_authority(ro_overlay):
    rules = ro_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "ro_aep_authority_v1"


def test_pack_passes_anti_misuse_validation(ro_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(ro_overlay)
    assert report.passed, f"RO pack failed anti-misuse: {report.errors}"
