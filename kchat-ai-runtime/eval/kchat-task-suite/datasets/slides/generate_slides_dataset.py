#!/usr/bin/env python3
"""Generate slides_eval_dataset_v1.json for the slides AI skill eval suite.

Produces ~1020 test cases:
- 210 templates × 4 cases each (EN short, EN long, multilingual, edge) = 840
- 12 deck-level cases (one per deck archetype × 10 languages... simplified to 12)
- 60 skill cases for the other 11 slides skills (suggest_template, rewrite, etc.)
"""

import json
import os

OUTPUT = os.path.join(os.path.dirname(__file__), "slides_eval_dataset_v1.json")

# Template IDs organized by family (must match slides_templates.rs)
TEMPLATES = {
    "Title": [
        "title", "title_subtitle", "title_image", "title_with_logo", "title_centered",
        "title_left", "cover_hero", "cover_split", "cover_video", "title_with_author",
        "title_with_date", "title_event",
    ],
    "Agenda": [
        "agenda", "agenda_numbered", "agenda_with_timing", "table_of_contents",
        "toc_two_col", "meeting_overview", "roadmap_agenda", "workshop_agenda",
        "agenda_linkable", "toc_with_page_numbers", "agenda_two_col",
        "agenda_with_breaks", "agenda_track", "agenda_with_speakers",
        "agenda_q_a", "agenda_with_goals",
    ],
    "Bullet": [
        "bullet", "bullet_with_icon", "numbered_list", "bullet_two_col",
        "bullet_three_col", "paragraph", "key_takeaway", "key_takeaway_with_image",
        "bullet_with_sub_bullets", "checklist", "definition_list", "bullet_with_image",
        "callout_box", "sticky_note", "bullet_pill", "bullet_arrow",
        "bullet_chevron", "bullet_star", "text_highlight", "bullet_with_stat",
    ],
    "Quote": [
        "quote", "quote_with_image", "pull_quote", "quote_large", "quote_with_author",
        "quote_centered", "quote_left", "testimonial_quote", "quote_with_background",
        "quote_block",
    ],
    "Comparison": [
        "comparison_two_col", "comparison_three_col", "comparison_four_col",
        "pros_cons", "versus", "before_after", "comparison_with_image",
        "comparison_table", "feature_matrix", "pricing_comparison", "comparison_split",
        "comparison_overlap", "comparison_stack", "comparison_pill", "comparison_arrow",
        "comparison_scale", "comparison_balance", "comparison_slider",
    ],
    "Timeline": [
        "timeline_horizontal", "timeline_vertical", "timeline_circular",
        "timeline_zigzag", "milestone", "milestone_with_image", "roadmap",
        "roadmap_with_phases", "gantt_summary", "process_timeline",
        "event_timeline", "history_timeline", "timeline_with_dates",
        "timeline_with_icons", "timeline_reverse", "timeline_convergent",
    ],
    "Image": [
        "image_full_bleed", "image_grid_2x2", "image_grid_3x3", "image_grid_4x4",
        "image_with_caption", "image_left_text_right", "image_right_text_left",
        "image_top_text_bottom", "image_bottom_text_top", "photo_collage",
        "hero_image", "image_stack", "image_with_overlay", "image_with_quote",
        "image_with_stat", "image_pair", "image_triple", "image_with_bullet",
        "image_centered", "image_with_attribution", "image_with_logo",
        "image_background_text", "image_gallery", "image_carousel",
    ],
    "Chart": [
        "bar_chart", "bar_chart_grouped", "bar_chart_stacked", "line_chart",
        "line_chart_multi", "pie_chart", "donut_chart", "stat_big_number",
        "stat_big_number_with_label", "kpi_dashboard", "kpi_grid_2", "kpi_grid_3",
        "kpi_grid_4", "data_table", "progress_bar", "progress_circle",
        "funnel", "funnel_with_labels", "gauge", "area_chart", "scatter_plot",
        "histogram",
    ],
    "Diagram": [
        "flowchart", "flowchart_vertical", "process_steps", "process_steps_circular",
        "pyramid", "pyramid_inverted", "venn_diagram", "venn_three", "swot",
        "puzzle", "puzzle_4", "hexagon", "hexagon_grid", "diamond",
        "circle_segments", "loop_cycle", "cycle_3", "cycle_4", "cycle_5",
        "matrix_2x2", "ladder", "iceberg", "iceberg_with_layers", "ecosystem",
    ],
    "Team": [
        "team_grid", "team_grid_2x2", "team_grid_3x3", "org_chart",
        "org_chart_two_level", "person_card", "person_card_with_image",
        "testimonial", "testimonial_with_image", "avatar_list", "speaker_bio",
        "speaker_bio_with_image", "team_with_roles", "leadership_team",
    ],
    "Media": [
        "video_embed", "icon_list", "icon_grid", "word_cloud",
        "word_cloud_weighted", "map", "map_with_pins", "qr_code",
        "audio_player", "social_feed", "screenshot_with_annotation", "code_block",
    ],
    "Section": [
        "section_break", "section_number", "section_icon", "divider_quote",
        "divider_full_bleed", "divider_gradient", "divider_with_logo", "transition",
        "recap", "next_steps", "closing", "thank_you", "qa_section", "resources",
        "appendix", "glossary", "references", "contact", "credits", "summary",
        "call_to_action", "discussion",
    ],
}

LANGUAGES = ["en", "vi", "ja", "zh", "es"]

MULTILINGUAL_BRIEFS = {
    "en": "Quarterly business review for Q3 2026",
    "vi": "Đánh giá kinh doanh quý 3 năm 2026",
    "ja": "2026年第3四半期のビジネスレビュー",
    "zh": "2026年第三季度业务回顾",
    "es": "Revisión comercial del tercer trimestre de 2026",
}

EDGE_BRIEFS = [
    "",  # empty brief
    "a",  # minimal
    "A comprehensive deep-dive analysis covering multiple strategic initiatives, market trends, competitive landscape, financial performance metrics, operational efficiency improvements, customer satisfaction surveys, employee engagement programs, and forward-looking roadmap considerations for the next fiscal year",  # maximal
]


def make_template_case(template_id, family, variant, brief, tier="medium"):
    """Create a single template test case."""
    case_id = f"slide_{template_id}_{variant}"
    # Quality checks depend on the family
    checks = [
        {"type": "min_length", "min_chars": 10},
        {"type": "coherent"},
        {"type": "json_schema_valid"},
    ]

    # Add template_conformance check
    checks.append({
        "type": "template_conformance",
        "template_id": template_id,
    })

    # Family-specific checks
    if family in ("Bullet", "Agenda"):
        checks.append({"type": "bullet_count", "min_bullets": 2, "max_bullets": 8})
    if family == "Chart":
        checks.append({"type": "chart_data_valid"})
    if family == "Image":
        checks.append({"type": "image_query_valid"})
    if family == "Quote":
        checks.append({"type": "min_length", "min_chars": 20})

    return {
        "id": case_id,
        "skill_id": "slides_generate_slide",
        "surface": "slides",
        "scope": "topic",
        "mode": "prompt_input",
        "input": {
            "document": "",
            "selection": "",
            "cursor_context": "",
            "variant_context": brief,
            "keywords": "",
        },
        "variant": None,
        "max_tokens": 512,
        "grammar_type": "json_schema",
        "quality_checks": checks,
        "expected_properties": {"template_id": template_id, "family": family},
        "tier": tier,
        "description": f"{family} template '{template_id}' — {variant} variant",
    }


def make_deck_case(archetype, lang, brief):
    """Create a deck-level test case."""
    case_id = f"deck_{archetype}_{lang}"
    checks = [
        {"type": "min_length", "min_chars": 50},
        {"type": "coherent"},
        {"type": "json_schema_valid"},
        {"type": "slot_count", "min_slots": 3, "max_slides": 15},
    ]
    return {
        "id": case_id,
        "skill_id": "slides_generate_deck",
        "surface": "slides",
        "scope": "topic",
        "mode": "multi_step",
        "input": {
            "document": "",
            "selection": "",
            "cursor_context": "",
            "variant_context": brief,
            "keywords": "",
        },
        "variant": None,
        "max_tokens": 1024,
        "grammar_type": "json_schema",
        "quality_checks": checks,
        "expected_properties": {"archetype": archetype, "language": lang},
        "tier": "high",
        "description": f"Deck generation — {archetype} ({lang})",
    }


def make_skill_case(skill_id, variant, input_text, context, checks, tier="medium"):
    """Create a skill-level test case."""
    case_id = f"skill_{skill_id}_{variant}"
    # Determine grammar_type from skill_id (must match SkillDef.grammar_type)
    if skill_id in ("slides_suggest_title",):
        grammar_type = "regex"
    elif skill_id in ("slides_summarize_deck", "slides_extract_speaker_notes",
                       "slides_translate_deck", "slides_key_takeaways"):
        grammar_type = "free_text"
    else:
        grammar_type = "json_schema"
    return {
        "id": case_id,
        "skill_id": skill_id,
        "surface": "slides",
        "scope": "selection" if "rewrite" in skill_id or "improve" in skill_id or "add_image" in skill_id else "document" if "summarize" in skill_id or "speaker" in skill_id or "translate" in skill_id or "takeaways" in skill_id else "topic",
        "mode": "one_click" if "suggest_template" in skill_id or "improve" in skill_id or "summarize" in skill_id or "speaker" in skill_id or "suggest_title" in skill_id or "takeaways" in skill_id else "prompt_input",
        "input": {
            "document": context,
            "selection": input_text if "rewrite" in skill_id or "improve" in skill_id or "add_image" in skill_id else "",
            "cursor_context": "",
            "variant_context": input_text if "topic" in skill_id or "suggest" in skill_id or "outline" in skill_id else "",
            "keywords": "",
        },
        "variant": None,
        "max_tokens": 400,
        "grammar_type": grammar_type,
        "quality_checks": checks,
        "expected_properties": {},
        "tier": tier,
        "description": f"Skill '{skill_id}' — {variant}",
    }


def generate():
    cases = []

    # 1. Template cases: 210 templates × 4 variants = 840
    for family, tmpl_ids in TEMPLATES.items():
        for tmpl_id in tmpl_ids:
            # EN short
            cases.append(make_template_case(tmpl_id, family, "en_short",
                f"Create a {family.lower()} slide about teamwork", "medium"))
            # EN long
            cases.append(make_template_case(tmpl_id, family, "en_long",
                f"Create a {family.lower()} slide about the importance of cross-functional collaboration in modern distributed teams", "high"))
            # Multilingual (rotate through languages)
            lang = LANGUAGES[len(cases) % len(LANGUAGES)]
            cases.append(make_template_case(tmpl_id, family, f"ml_{lang}",
                MULTILINGUAL_BRIEFS[lang], "medium"))
            # Edge case
            edge_brief = EDGE_BRIEFS[len(cases) % len(EDGE_BRIEFS)]
            cases.append(make_template_case(tmpl_id, family, "edge",
                edge_brief, "low"))

    # 2. Deck-level cases: 12 archetypes
    archetypes = [
        ("pitch", "Investor pitch deck for a SaaS startup"),
        ("report", "Annual report presentation"),
        ("training", "Employee onboarding training deck"),
        ("webinar", "Product launch webinar slides"),
        ("workshop", "Design thinking workshop deck"),
        ("conference", "Conference talk on AI ethics"),
        ("sales", "Sales enablement deck for Q4"),
        ("strategy", "Strategic planning offsite deck"),
        ("review", "Quarterly business review"),
        ("proposal", "Project proposal presentation"),
        ("education", "Course lecture on machine learning"),
        ("marketing", "Brand marketing campaign deck"),
    ]
    for archetype, brief in archetypes:
        cases.append(make_deck_case(archetype, "en", brief))

    # 3. Skill cases for the other 11 slides skills (60 total)
    skill_cases = [
        ("slides_suggest_template", "01", "A comparison of two product features", "",
         [{"type": "json_schema_valid"}, {"type": "template_conformance"}], "low"),
        ("slides_suggest_template", "02", "Show quarterly revenue growth", "",
         [{"type": "json_schema_valid"}, {"type": "template_conformance"}], "low"),
        ("slides_suggest_template", "03", "Introduce the keynote speaker", "",
         [{"type": "json_schema_valid"}, {"type": "template_conformance"}], "low"),
        ("slides_suggest_template", "04", "Display the project timeline", "",
         [{"type": "json_schema_valid"}, {"type": "template_conformance"}], "low"),
        ("slides_suggest_template", "05", "Highlight customer testimonials", "",
         [{"type": "json_schema_valid"}, {"type": "template_conformance"}], "low"),

        ("slides_suggest_outline", "01", "AI in healthcare", "",
         [{"type": "json_schema_valid"}, {"type": "min_length", "min_chars": 50}], "medium"),
        ("slides_suggest_outline", "02", "Climate change solutions", "",
         [{"type": "json_schema_valid"}, {"type": "min_length", "min_chars": 50}], "medium"),
        ("slides_suggest_outline", "03", "Digital transformation strategy", "",
         [{"type": "json_schema_valid"}, {"type": "min_length", "min_chars": 50}], "medium"),

        ("slides_rewrite_slide", "01", "Our company makes software that helps people.", "Make it more impactful",
         [{"type": "json_schema_valid"}, {"type": "min_length", "min_chars": 20}], "medium"),
        ("slides_rewrite_slide", "02", "We are excited to announce our new product.", "Make it punchier",
         [{"type": "json_schema_valid"}, {"type": "min_length", "min_chars": 20}], "medium"),
        ("slides_rewrite_slide", "03", "The results were good this quarter.", "Make it data-driven",
         [{"type": "json_schema_valid"}, {"type": "min_length", "min_chars": 20}], "medium"),

        ("slides_improve_slide", "01", "This slide is about how our team works together to achieve great things for our customers.", "",
         [{"type": "json_schema_valid"}, {"type": "max_length", "max_chars": 200}], "low"),
        ("slides_improve_slide", "02", "We have many features including analytics, reporting, dashboards, integrations, and more.", "",
         [{"type": "json_schema_valid"}, {"type": "max_length", "max_chars": 200}], "low"),
        ("slides_improve_slide", "03", "The presentation will cover an introduction to our topic and then we will discuss the main points.", "",
         [{"type": "json_schema_valid"}, {"type": "max_length", "max_chars": 200}], "low"),

        ("slides_add_image", "01", "A slide about renewable energy", "",
         [{"type": "json_schema_valid"}, {"type": "image_query_valid"}], "low"),
        ("slides_add_image", "02", "A slide about team collaboration", "",
         [{"type": "json_schema_valid"}, {"type": "image_query_valid"}], "low"),
        ("slides_add_image", "03", "A slide about global markets", "",
         [{"type": "json_schema_valid"}, {"type": "image_query_valid"}], "low"),

        ("slides_summarize_deck", "01", "",
         "Slide 1: Introduction. Slide 2: Market Overview. Slide 3: Product Features. Slide 4: Financial Projections. Slide 5: Next Steps.",
         [{"type": "min_length", "min_chars": 50}, {"type": "coherent"}, {"type": "sentence_count", "min_sentences": 2, "max_sentences": 8}], "medium"),
        ("slides_summarize_deck", "02", "",
         "Slide 1: Q3 Results. Slide 2: Revenue Growth. Slide 3: Customer Acquisition. Slide 4: Challenges. Slide 5: Q4 Plans.",
         [{"type": "min_length", "min_chars": 50}, {"type": "coherent"}, {"type": "sentence_count", "min_sentences": 2, "max_sentences": 8}], "medium"),

        ("slides_extract_speaker_notes", "01", "",
         "Slide 1: Company Mission. Slide 2: Core Values. Slide 3: Team Structure.",
         [{"type": "min_length", "min_chars": 50}, {"type": "coherent"}], "medium"),
        ("slides_extract_speaker_notes", "02", "",
         "Slide 1: Problem Statement. Slide 2: Solution. Slide 3: Market Size. Slide 4: Business Model.",
         [{"type": "min_length", "min_chars": 50}, {"type": "coherent"}], "medium"),

        ("slides_translate_deck", "01", "",
         "Slide 1: Welcome. Slide 2: Agenda. Slide 3: Key Findings. Slide 4: Conclusion.",
         [{"type": "min_length", "min_chars": 30}, {"type": "language_script", "language": "spanish"}], "medium"),
        ("slides_translate_deck", "02", "",
         "Slide 1: Introduction. Slide 2: Methodology. Slide 3: Results. Slide 4: Discussion.",
         [{"type": "min_length", "min_chars": 30}, {"type": "language_script", "language": "french"}], "medium"),

        ("slides_suggest_title", "01", "AI in modern education", "",
         [{"type": "min_length", "min_chars": 3}, {"type": "max_length", "max_chars": 80}], "low"),
        ("slides_suggest_title", "02", "Sustainable business practices", "",
         [{"type": "min_length", "min_chars": 3}, {"type": "max_length", "max_chars": 80}], "low"),
        ("slides_suggest_title", "03", "The future of remote work", "",
         [{"type": "min_length", "min_chars": 3}, {"type": "max_length", "max_chars": 80}], "low"),

        ("slides_key_takeaways", "01", "",
         "Slide 1: Revenue up 20%. Slide 2: New product launched. Slide 3: Expanded to 3 markets. Slide 4: Hiring 50 people.",
         [{"type": "min_length", "min_chars": 30}, {"type": "coherent"}, {"type": "sentence_count", "min_sentences": 3, "max_sentences": 6}], "low"),
        ("slides_key_takeaways", "02", "",
         "Slide 1: Customer satisfaction at 95%. Slide 2: Churn reduced by 15%. Slide 3: NPS score of 72. Slide 4: 3 new enterprise clients.",
         [{"type": "min_length", "min_chars": 30}, {"type": "coherent"}, {"type": "sentence_count", "min_sentences": 3, "max_sentences": 6}], "low"),
    ]

    for sc in skill_cases:
        cases.append(make_skill_case(*sc))

    dataset = {
        "name": "kchat-slides-eval-v1",
        "version": "1.0.0",
        "description": f"Per-skill quality eval dataset for all 12 slides AI skills across 210 smart templates. {len(cases)} test cases with realistic inputs, quality checks, and expected properties.",
        "test_cases": cases,
    }

    with open(OUTPUT, "w") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(cases)} test cases → {OUTPUT}")
    # Verify counts
    template_cases = sum(1 for c in cases if c["skill_id"] == "slides_generate_slide")
    deck_cases = sum(1 for c in cases if c["skill_id"] == "slides_generate_deck")
    other_cases = len(cases) - template_cases - deck_cases
    print(f"  Template cases: {template_cases}")
    print(f"  Deck cases: {deck_cases}")
    print(f"  Other skill cases: {other_cases}")


if __name__ == "__main__":
    generate()
