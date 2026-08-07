"""Validate the kchat.jurisdiction.ec.guardrail.v1 country pack.

Country-specific overrides
match the overlay's docstring:
  Codigo de la Ninez y Adolescencia (CONA); COIP Art. 366 anti-terrorism.
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "ec"


def test_run_all_structural_assertions(ec_overlay, ec_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        ec_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=ec_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(ec_overlay):
    override = A.get_override(ec_overlay, 1)
    assert override["severity_floor"] == 5, (
        "EC: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_4(ec_overlay):
    override = A.get_override(ec_overlay, 4)
    assert override["severity_floor"] == 4, (
        "EC: category 4 override must be severity_floor 4"
    )


def test_legal_age_marketplace_alcohol_is_18(ec_overlay):
    assert ec_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 18


def test_legal_age_marketplace_tobacco_is_18(ec_overlay):
    assert ec_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 18


def test_protected_classes_includes_statutory_enumeration(ec_overlay):
    classes = set(ec_overlay["local_definitions"]["protected_classes"])
    expected = {'ethnic_origin', 'sex', 'religion', 'race'}
    assert expected <= classes, (
        "EC protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(ec_overlay):
    langs = list(ec_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['es'], (
        f"EC primary_languages must equal ['es']; got {langs}"
    )


def test_user_notice_opt_out_allowed_true(ec_overlay):
    assert ec_overlay["user_notice"]["opt_out_allowed"] is True


def test_election_rules_reference_authority(ec_overlay):
    rules = ec_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "ec_cne_authority_v1"


def test_pack_passes_anti_misuse_validation(ec_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(ec_overlay)
    assert report.passed, f"EC pack failed anti-misuse: {report.errors}"
