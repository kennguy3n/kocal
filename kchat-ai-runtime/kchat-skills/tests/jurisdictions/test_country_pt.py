"""Validate the kchat.jurisdiction.pt.guardrail.v1 country pack.

Country-specific overrides match
the overlay's docstring:
  Código Penal child protection; Lei 52/2003 anti-terrorismo.
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "pt"


def test_run_all_structural_assertions(pt_overlay, pt_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        pt_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=pt_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(pt_overlay):
    override = A.get_override(pt_overlay, 1)
    assert override["severity_floor"] == 5, (
        "PT: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_4(pt_overlay):
    override = A.get_override(pt_overlay, 4)
    assert override["severity_floor"] == 4, (
        "PT: category 4 override must be severity_floor 4"
    )


def test_legal_age_marketplace_alcohol_is_18(pt_overlay):
    assert pt_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 18


def test_legal_age_marketplace_tobacco_is_18(pt_overlay):
    assert pt_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 18


def test_protected_classes_includes_statutory_enumeration(pt_overlay):
    classes = set(pt_overlay["local_definitions"]["protected_classes"])
    expected = {'sex', 'race', 'religion', 'disability', 'nationality'}
    assert expected <= classes, (
        "PT protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(pt_overlay):
    langs = list(pt_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['pt'], (
        f"PT primary_languages must equal ['pt']; got {langs}"
    )


def test_user_notice_opt_out_allowed_true(pt_overlay):
    assert pt_overlay["user_notice"]["opt_out_allowed"] is True


def test_election_rules_reference_authority(pt_overlay):
    rules = pt_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "pt_cne_authority_v1"


def test_pack_passes_anti_misuse_validation(pt_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(pt_overlay)
    assert report.passed, f"PT pack failed anti-misuse: {report.errors}"
