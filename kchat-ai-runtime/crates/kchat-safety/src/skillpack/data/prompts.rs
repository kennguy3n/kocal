//! Embedded prompt templates and compiled prompt examples.
//!
//! These files are embedded from `files/prompts/`.

/// `files/prompts/runtime_instruction.txt` — runtime instruction template.
pub const RUNTIME_INSTRUCTION_TXT: &str =
    include_str!("files/prompts/runtime_instruction.txt");

/// `files/prompts/compiled_prompt_format.md` — compiled prompt format documentation.
pub const COMPILED_PROMPT_FORMAT_MD: &str =
    include_str!("files/prompts/compiled_prompt_format.md");

/// Get a compiled prompt example by filename (without `.txt` extension).
///
/// Returns `None` if the filename doesn't match any embedded example.
pub fn compiled_prompt(name: &str) -> Option<&'static str> {
    match name {
        "baseline_only" => Some(include_str!(
            "files/prompts/compiled_examples/baseline_only.txt"
        )),
        "community_adult_only" => Some(include_str!(
            "files/prompts/compiled_examples/community_adult_only.txt"
        )),
        "community_family" => Some(include_str!(
            "files/prompts/compiled_examples/community_family.txt"
        )),
        "community_gaming" => Some(include_str!(
            "files/prompts/compiled_examples/community_gaming.txt"
        )),
        "community_health_support" => Some(include_str!(
            "files/prompts/compiled_examples/community_health_support.txt"
        )),
        "community_marketplace" => Some(include_str!(
            "files/prompts/compiled_examples/community_marketplace.txt"
        )),
        "community_political" => Some(include_str!(
            "files/prompts/compiled_examples/community_political.txt"
        )),
        "community_school" => Some(include_str!(
            "files/prompts/compiled_examples/community_school.txt"
        )),
        "community_workplace" => Some(include_str!(
            "files/prompts/compiled_examples/community_workplace.txt"
        )),
        "jurisdiction_strict_adult" => Some(include_str!(
            "files/prompts/compiled_examples/jurisdiction_strict_adult.txt"
        )),
        "jurisdiction_strict_hate" => Some(include_str!(
            "files/prompts/compiled_examples/jurisdiction_strict_hate.txt"
        )),
        "jurisdiction_strict_marketplace" => Some(include_str!(
            "files/prompts/compiled_examples/jurisdiction_strict_marketplace.txt"
        )),
        "strict_adult_school" => Some(include_str!(
            "files/prompts/compiled_examples/strict_adult_school.txt"
        )),
        "strict_marketplace_workplace" => Some(include_str!(
            "files/prompts/compiled_examples/strict_marketplace_workplace.txt"
        )),
        // Country-specific compiled prompts
        "country_ae" => Some(include_str!(
            "files/prompts/compiled_examples/country_ae.txt"
        )),
        "country_ar" => Some(include_str!(
            "files/prompts/compiled_examples/country_ar.txt"
        )),
        "country_at" => Some(include_str!(
            "files/prompts/compiled_examples/country_at.txt"
        )),
        "country_au" => Some(include_str!(
            "files/prompts/compiled_examples/country_au.txt"
        )),
        "country_bd" => Some(include_str!(
            "files/prompts/compiled_examples/country_bd.txt"
        )),
        "country_br" => Some(include_str!(
            "files/prompts/compiled_examples/country_br.txt"
        )),
        "country_ca" => Some(include_str!(
            "files/prompts/compiled_examples/country_ca.txt"
        )),
        "country_ch" => Some(include_str!(
            "files/prompts/compiled_examples/country_ch.txt"
        )),
        "country_cl" => Some(include_str!(
            "files/prompts/compiled_examples/country_cl.txt"
        )),
        "country_co" => Some(include_str!(
            "files/prompts/compiled_examples/country_co.txt"
        )),
        "country_cz" => Some(include_str!(
            "files/prompts/compiled_examples/country_cz.txt"
        )),
        "country_de" => Some(include_str!(
            "files/prompts/compiled_examples/country_de.txt"
        )),
        "country_dk" => Some(include_str!(
            "files/prompts/compiled_examples/country_dk.txt"
        )),
        "country_dz" => Some(include_str!(
            "files/prompts/compiled_examples/country_dz.txt"
        )),
        "country_ec" => Some(include_str!(
            "files/prompts/compiled_examples/country_ec.txt"
        )),
        "country_eg" => Some(include_str!(
            "files/prompts/compiled_examples/country_eg.txt"
        )),
        "country_es" => Some(include_str!(
            "files/prompts/compiled_examples/country_es.txt"
        )),
        "country_et" => Some(include_str!(
            "files/prompts/compiled_examples/country_et.txt"
        )),
        "country_fi" => Some(include_str!(
            "files/prompts/compiled_examples/country_fi.txt"
        )),
        "country_fr" => Some(include_str!(
            "files/prompts/compiled_examples/country_fr.txt"
        )),
        "country_gb" => Some(include_str!(
            "files/prompts/compiled_examples/country_gb.txt"
        )),
        "country_gh" => Some(include_str!(
            "files/prompts/compiled_examples/country_gh.txt"
        )),
        "country_gr" => Some(include_str!(
            "files/prompts/compiled_examples/country_gr.txt"
        )),
        "country_hu" => Some(include_str!(
            "files/prompts/compiled_examples/country_hu.txt"
        )),
        "country_id" => Some(include_str!(
            "files/prompts/compiled_examples/country_id.txt"
        )),
        "country_ie" => Some(include_str!(
            "files/prompts/compiled_examples/country_ie.txt"
        )),
        "country_il" => Some(include_str!(
            "files/prompts/compiled_examples/country_il.txt"
        )),
        "country_in" => Some(include_str!(
            "files/prompts/compiled_examples/country_in.txt"
        )),
        "country_iq" => Some(include_str!(
            "files/prompts/compiled_examples/country_iq.txt"
        )),
        "country_it" => Some(include_str!(
            "files/prompts/compiled_examples/country_it.txt"
        )),
        "country_jp" => Some(include_str!(
            "files/prompts/compiled_examples/country_jp.txt"
        )),
        "country_ke" => Some(include_str!(
            "files/prompts/compiled_examples/country_ke.txt"
        )),
        "country_kr" => Some(include_str!(
            "files/prompts/compiled_examples/country_kr.txt"
        )),
        "country_ma" => Some(include_str!(
            "files/prompts/compiled_examples/country_ma.txt"
        )),
        "country_mx" => Some(include_str!(
            "files/prompts/compiled_examples/country_mx.txt"
        )),
        "country_my" => Some(include_str!(
            "files/prompts/compiled_examples/country_my.txt"
        )),
        "country_ng" => Some(include_str!(
            "files/prompts/compiled_examples/country_ng.txt"
        )),
        "country_nl" => Some(include_str!(
            "files/prompts/compiled_examples/country_nl.txt"
        )),
        "country_no" => Some(include_str!(
            "files/prompts/compiled_examples/country_no.txt"
        )),
        "country_nz" => Some(include_str!(
            "files/prompts/compiled_examples/country_nz.txt"
        )),
        "country_pe" => Some(include_str!(
            "files/prompts/compiled_examples/country_pe.txt"
        )),
        "country_ph" => Some(include_str!(
            "files/prompts/compiled_examples/country_ph.txt"
        )),
        "country_pk" => Some(include_str!(
            "files/prompts/compiled_examples/country_pk.txt"
        )),
        "country_pl" => Some(include_str!(
            "files/prompts/compiled_examples/country_pl.txt"
        )),
        "country_pt" => Some(include_str!(
            "files/prompts/compiled_examples/country_pt.txt"
        )),
        "country_ro" => Some(include_str!(
            "files/prompts/compiled_examples/country_ro.txt"
        )),
        "country_ru" => Some(include_str!(
            "files/prompts/compiled_examples/country_ru.txt"
        )),
        "country_sa" => Some(include_str!(
            "files/prompts/compiled_examples/country_sa.txt"
        )),
        "country_se" => Some(include_str!(
            "files/prompts/compiled_examples/country_se.txt"
        )),
        "country_sg" => Some(include_str!(
            "files/prompts/compiled_examples/country_sg.txt"
        )),
        "country_th" => Some(include_str!(
            "files/prompts/compiled_examples/country_th.txt"
        )),
        "country_tr" => Some(include_str!(
            "files/prompts/compiled_examples/country_tr.txt"
        )),
        "country_tw" => Some(include_str!(
            "files/prompts/compiled_examples/country_tw.txt"
        )),
        "country_tz" => Some(include_str!(
            "files/prompts/compiled_examples/country_tz.txt"
        )),
        "country_ua" => Some(include_str!(
            "files/prompts/compiled_examples/country_ua.txt"
        )),
        "country_us" => Some(include_str!(
            "files/prompts/compiled_examples/country_us.txt"
        )),
        "country_uy" => Some(include_str!(
            "files/prompts/compiled_examples/country_uy.txt"
        )),
        "country_vn" => Some(include_str!(
            "files/prompts/compiled_examples/country_vn.txt"
        )),
        "country_za" => Some(include_str!(
            "files/prompts/compiled_examples/country_za.txt"
        )),
        _ => None,
    }
}

/// List all compiled prompt example names.
pub fn compiled_prompt_names() -> &'static [&'static str] {
    &[
        "baseline_only",
        "community_adult_only",
        "community_family",
        "community_gaming",
        "community_health_support",
        "community_marketplace",
        "community_political",
        "community_school",
        "community_workplace",
        "jurisdiction_strict_adult",
        "jurisdiction_strict_hate",
        "jurisdiction_strict_marketplace",
        "strict_adult_school",
        "strict_marketplace_workplace",
        "country_ae", "country_ar", "country_at", "country_au", "country_bd",
        "country_br", "country_ca", "country_ch", "country_cl", "country_co",
        "country_cz", "country_de", "country_dk", "country_dz", "country_ec",
        "country_eg", "country_es", "country_et", "country_fi", "country_fr",
        "country_gb", "country_gh", "country_gr", "country_hu", "country_id",
        "country_ie", "country_il", "country_in", "country_iq", "country_it",
        "country_jp", "country_ke", "country_kr", "country_ma", "country_mx",
        "country_my", "country_ng", "country_nl", "country_no", "country_nz",
        "country_pe", "country_ph", "country_pk", "country_pl", "country_pt",
        "country_ro", "country_ru", "country_sa", "country_se", "country_sg",
        "country_th", "country_tr", "country_tw", "country_tz", "country_ua",
        "country_us", "country_uy", "country_vn", "country_za",
    ]
}
