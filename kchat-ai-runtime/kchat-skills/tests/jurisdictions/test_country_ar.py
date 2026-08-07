"""Validate the kchat.jurisdiction.ar.guardrail.v1 country pack.

Country-specific overrides match
the overlay's docstring:
  Ley 26.061 child protection; Código Penal terrorism provisions.
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "ar"


def test_run_all_structural_assertions(ar_overlay, ar_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        ar_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=ar_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(ar_overlay):
    override = A.get_override(ar_overlay, 1)
    assert override["severity_floor"] == 5, (
        "AR: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_4(ar_overlay):
    override = A.get_override(ar_overlay, 4)
    assert override["severity_floor"] == 4, (
        "AR: category 4 override must be severity_floor 4"
    )


def test_legal_age_marketplace_alcohol_is_18(ar_overlay):
    assert ar_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 18


def test_legal_age_marketplace_tobacco_is_18(ar_overlay):
    assert ar_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 18


def test_protected_classes_includes_statutory_enumeration(ar_overlay):
    classes = set(ar_overlay["local_definitions"]["protected_classes"])
    expected = {'sex', 'race', 'religion', 'disability', 'nationality'}
    assert expected <= classes, (
        "AR protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(ar_overlay):
    langs = list(ar_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['es'], (
        f"AR primary_languages must equal ['es']; got {langs}"
    )


def test_user_notice_opt_out_allowed_true(ar_overlay):
    assert ar_overlay["user_notice"]["opt_out_allowed"] is True


def test_election_rules_reference_authority(ar_overlay):
    rules = ar_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "ar_cne_authority_v1"


def test_pack_passes_anti_misuse_validation(ar_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(ar_overlay)
    assert report.passed, f"AR pack failed anti-misuse: {report.errors}"
