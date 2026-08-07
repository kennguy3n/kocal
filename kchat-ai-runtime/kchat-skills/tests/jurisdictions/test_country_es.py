"""Validate the kchat.jurisdiction.es.guardrail.v1 country pack.

Country-specific overrides match
the overlay's docstring:
  Ley Orgánica de Protección del Menor; Código Penal Art. 571-580 (terrorismo).
"""
from __future__ import annotations

from . import _country_pack_assertions as A

COUNTRY_CODE = "es"


def test_run_all_structural_assertions(es_overlay, es_normalization, jurisdictions_dir):
    A.run_all_structural_assertions(
        es_overlay,
        country_code=COUNTRY_CODE,
        normalization_file=es_normalization,
        jurisdictions_dir=jurisdictions_dir,
    )


def test_category_1_child_safety_floor_5(es_overlay):
    override = A.get_override(es_overlay, 1)
    assert override["severity_floor"] == 5, (
        "ES: category 1 override must be severity_floor 5"
    )


def test_category_4_extremism_floor_4(es_overlay):
    override = A.get_override(es_overlay, 4)
    assert override["severity_floor"] == 4, (
        "ES: category 4 override must be severity_floor 4"
    )


def test_legal_age_marketplace_alcohol_is_18(es_overlay):
    assert es_overlay["local_definitions"]["legal_age_marketplace_alcohol"] == 18


def test_legal_age_marketplace_tobacco_is_18(es_overlay):
    assert es_overlay["local_definitions"]["legal_age_marketplace_tobacco"] == 18


def test_protected_classes_includes_statutory_enumeration(es_overlay):
    classes = set(es_overlay["local_definitions"]["protected_classes"])
    expected = {'sex', 'race', 'ethnic_origin', 'religion', 'disability'}
    assert expected <= classes, (
        "ES protected_classes must include the statutory "
        f"enumeration; missing: {expected - classes}"
    )


def test_primary_languages(es_overlay):
    langs = list(es_overlay["local_language_assets"]["primary_languages"])
    assert langs == ['es', 'ca', 'eu', 'gl'], (
        f"ES primary_languages must equal ['es', 'ca', 'eu', 'gl']; got {langs}"
    )


def test_user_notice_opt_out_allowed_true(es_overlay):
    assert es_overlay["user_notice"]["opt_out_allowed"] is True


def test_election_rules_reference_authority(es_overlay):
    rules = es_overlay["local_definitions"]["election_rules"]
    assert rules["authority_resource_id"] == "es_jec_authority_v1"


def test_pack_passes_anti_misuse_validation(es_overlay):
    from compiler.anti_misuse import validate_pack  # type: ignore

    report = validate_pack(es_overlay)
    assert report.passed, f"ES pack failed anti-misuse: {report.errors}"
