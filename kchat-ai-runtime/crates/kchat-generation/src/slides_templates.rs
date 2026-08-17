//! Smart slide templates — beautiful.ai-inspired declarative template catalog.
//!
//! Each `SlidesTemplate` declares its family, slots (typed placeholders),
//! layout hints, image orientation hint, and a JSON Schema used for
//! grammar-constrained generation. The model emits slot fill JSON that
//! conforms to the template's schema; the runtime renders it.
//!
//! 210 templates across 12 families: Title, Agenda, Bullet, Quote,
//! Comparison, Timeline, Image, Chart, Diagram, Team, Media, Section.

use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// One of 12 smart slide layout families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SlidesTemplateFamily {
    Title,
    Agenda,
    Bullet,
    Quote,
    Comparison,
    Timeline,
    Image,
    Chart,
    Diagram,
    Team,
    Media,
    Section,
}

impl SlidesTemplateFamily {
    pub fn label(self) -> &'static str {
        match self {
            SlidesTemplateFamily::Title => "Title",
            SlidesTemplateFamily::Agenda => "Agenda",
            SlidesTemplateFamily::Bullet => "Bullet",
            SlidesTemplateFamily::Quote => "Quote",
            SlidesTemplateFamily::Comparison => "Comparison",
            SlidesTemplateFamily::Timeline => "Timeline",
            SlidesTemplateFamily::Image => "Image",
            SlidesTemplateFamily::Chart => "Chart",
            SlidesTemplateFamily::Diagram => "Diagram",
            SlidesTemplateFamily::Team => "Team",
            SlidesTemplateFamily::Media => "Media",
            SlidesTemplateFamily::Section => "Section",
        }
    }
}

/// Type of a slot within a slide template.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SlotType {
    TitleText,
    SubtitleText,
    BodyText,
    BulletList,
    ImageQuery,
    ImageRef,
    ChartSeries,
    StatNumber,
    StatLabel,
    QuoteText,
    AttributionText,
    PersonName,
    PersonRole,
    StepList,
    DateList,
    NumberedList,
    LabelText,
    CaptionText,
    FooterText,
    SectionLabel,
}

impl SlotType {
    /// JSON Schema fragment for this slot type (the value schema).
    fn value_schema(self) -> Value {
        match self {
            SlotType::TitleText | SlotType::SubtitleText | SlotType::BodyText
            | SlotType::QuoteText | SlotType::AttributionText | SlotType::PersonName
            | SlotType::PersonRole | SlotType::LabelText | SlotType::CaptionText
            | SlotType::FooterText | SlotType::SectionLabel | SlotType::StatLabel => {
                serde_json::json!({"type": "string"})
            }
            SlotType::BulletList | SlotType::StepList | SlotType::DateList
            | SlotType::NumberedList => {
                serde_json::json!({
                    "type": "array",
                    "items": {"type": "string"}
                })
            }
            SlotType::ImageQuery | SlotType::ImageRef => {
                serde_json::json!({
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": {"type": "string"},
                        "orientation": {"type": "string", "enum": ["landscape", "portrait", "square"]}
                    }
                })
            }
            SlotType::ChartSeries => {
                serde_json::json!({
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["label", "value"],
                        "properties": {
                            "label": {"type": "string"},
                            "value": {"type": "number"}
                        }
                    }
                })
            }
            SlotType::StatNumber => serde_json::json!({"type": "number"}),
        }
    }
}

/// A single slot definition within a template.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlotDef {
    pub id: String,
    pub slot_type: SlotType,
    pub label: String,
    pub required: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_items: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_items: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub placeholder: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hint: Option<String>,
}

impl SlotDef {
    fn new(id: &str, slot_type: SlotType, label: &str) -> Self {
        Self {
            id: id.into(),
            slot_type,
            label: label.into(),
            required: true,
            min_items: None,
            max_items: None,
            placeholder: None,
            hint: None,
        }
    }

    fn optional(mut self) -> Self {
        self.required = false;
        self
    }

    fn items(mut self, min: usize, max: usize) -> Self {
        self.min_items = Some(min);
        self.max_items = Some(max);
        self
    }

    fn placeholder(mut self, p: &str) -> Self {
        self.placeholder = Some(p.into());
        self
    }

    fn hint(mut self, h: &str) -> Self {
        self.hint = Some(h.into());
        self
    }
}

/// A smart slide template.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlidesTemplate {
    pub id: String,
    pub label: String,
    pub family: SlidesTemplateFamily,
    pub slots: Vec<SlotDef>,
    /// Layout hint for the renderer (e.g. "split_left", "grid_2x2").
    pub layout_hint: String,
    /// Preferred image orientation for image-bearing templates.
    pub image_orientation_hint: Option<&'static str>,
    /// Maximum number of bullets (for bullet-family templates).
    pub max_bullets: usize,
    /// Whether this template supports chart data.
    pub supports_chart: bool,
    /// Lucide icon name.
    pub icon: String,
}

impl SlidesTemplate {
    /// Build the JSON Schema for slot fill of this template.
    pub fn slot_schema(&self) -> Value {
        let mut props = serde_json::Map::new();
        let mut required: Vec<String> = Vec::new();
        for slot in &self.slots {
            props.insert(slot.id.clone(), slot.slot_type.value_schema());
            if slot.required {
                required.push(slot.id.clone());
            }
        }
        serde_json::json!({
            "type": "object",
            "required": required,
            "properties": props,
            "additionalProperties": false
        })
    }
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/// Registry of all smart slide templates.
pub struct SlidesTemplateRegistry {
    templates: Vec<SlidesTemplate>,
    index: HashMap<String, usize>,
}

/// Process-wide singleton registry — constructed once, reused forever.
/// Avoids re-allocating 210 templates on every prompt build or validation call.
pub static TEMPLATE_REGISTRY: Lazy<SlidesTemplateRegistry> = Lazy::new(SlidesTemplateRegistry::new);

/// Process-wide singleton catalog string — avoids rebuilding the 210-line
/// catalog on every prompt construction.
pub static TEMPLATE_CATALOG: Lazy<String> = Lazy::new(|| TEMPLATE_REGISTRY.compact_catalog());

impl SlidesTemplateRegistry {
    /// Create a registry pre-loaded with all 210 smart templates.
    pub fn new() -> Self {
        let templates = all_templates();
        let index: HashMap<String, usize> = templates
            .iter()
            .enumerate()
            .map(|(i, t)| (t.id.clone(), i))
            .collect();
        Self { templates, index }
    }

    /// O(1) lookup by template ID via HashMap index.
    pub fn get(&self, id: &str) -> Option<&SlidesTemplate> {
        self.index.get(id).map(|&i| &self.templates[i])
    }

    pub fn all(&self) -> &[SlidesTemplate] {
        &self.templates
    }

    pub fn by_family(&self, family: SlidesTemplateFamily) -> Vec<&SlidesTemplate> {
        self.templates
            .iter()
            .filter(|t| t.family == family)
            .collect()
    }

    pub fn len(&self) -> usize {
        self.templates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.templates.is_empty()
    }

    /// Compact catalog (id + label + family) for injection into prompts.
    pub fn compact_catalog(&self) -> String {
        self.templates
            .iter()
            .map(|t| format!("{}|{}|{}", t.id, t.label, t.family.label()))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl Default for SlidesTemplateRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Template construction helpers
// ---------------------------------------------------------------------------

fn t(id: &str, label: &str, family: SlidesTemplateFamily, slots: Vec<SlotDef>, layout: &str, icon: &str) -> SlidesTemplate {
    SlidesTemplate {
        id: id.into(),
        label: label.into(),
        family,
        slots,
        layout_hint: layout.into(),
        image_orientation_hint: None,
        max_bullets: 0,
        supports_chart: false,
        icon: icon.into(),
    }
}

fn title_slot() -> SlotDef { SlotDef::new("title", SlotType::TitleText, "Title") }
fn subtitle_slot() -> SlotDef { SlotDef::new("subtitle", SlotType::SubtitleText, "Subtitle").optional() }
fn body_slot() -> SlotDef { SlotDef::new("body", SlotType::BodyText, "Body") }
fn bullets_slot(min: usize, max: usize) -> SlotDef {
    SlotDef::new("bullets", SlotType::BulletList, "Bullets").items(min, max)
}
fn image_slot(orientation: &'static str) -> SlotDef {
    SlotDef::new("image", SlotType::ImageQuery, "Image").hint(orientation)
}
fn caption_slot() -> SlotDef { SlotDef::new("caption", SlotType::CaptionText, "Caption").optional() }
fn quote_slot() -> SlotDef { SlotDef::new("quote", SlotType::QuoteText, "Quote") }
fn attribution_slot() -> SlotDef { SlotDef::new("attribution", SlotType::AttributionText, "Attribution").optional() }
fn stat_number_slot() -> SlotDef { SlotDef::new("value", SlotType::StatNumber, "Value") }
fn stat_label_slot() -> SlotDef { SlotDef::new("label", SlotType::StatLabel, "Label") }
fn chart_series_slot() -> SlotDef { SlotDef::new("series", SlotType::ChartSeries, "Data Series") }
fn person_name_slot() -> SlotDef { SlotDef::new("name", SlotType::PersonName, "Name") }
fn person_role_slot() -> SlotDef { SlotDef::new("role", SlotType::PersonRole, "Role").optional() }
fn steps_slot(min: usize, max: usize) -> SlotDef {
    SlotDef::new("steps", SlotType::StepList, "Steps").items(min, max)
}
fn dates_slot(min: usize, max: usize) -> SlotDef {
    SlotDef::new("dates", SlotType::DateList, "Dates").items(min, max)
}
fn footer_slot() -> SlotDef { SlotDef::new("footer", SlotType::FooterText, "Footer").optional() }
fn section_label_slot() -> SlotDef { SlotDef::new("section_label", SlotType::SectionLabel, "Section Label") }
fn numbered_slot(min: usize, max: usize) -> SlotDef {
    SlotDef::new("items", SlotType::NumberedList, "Items").items(min, max)
}

fn with_image_orientation(mut tmpl: SlidesTemplate, o: &'static str) -> SlidesTemplate {
    tmpl.image_orientation_hint = Some(o);
    tmpl
}

fn with_max_bullets(mut tmpl: SlidesTemplate, n: usize) -> SlidesTemplate {
    tmpl.max_bullets = n;
    tmpl
}

fn with_chart(mut tmpl: SlidesTemplate) -> SlidesTemplate {
    tmpl.supports_chart = true;
    tmpl
}

// ---------------------------------------------------------------------------
// All 210 templates
// ---------------------------------------------------------------------------

fn all_templates() -> Vec<SlidesTemplate> {
    let mut out: Vec<SlidesTemplate> = Vec::with_capacity(210);
    out.extend(title_family());
    out.extend(agenda_family());
    out.extend(bullet_family());
    out.extend(quote_family());
    out.extend(comparison_family());
    out.extend(timeline_family());
    out.extend(image_family());
    out.extend(chart_family());
    out.extend(diagram_family());
    out.extend(team_family());
    out.extend(media_family());
    out.extend(section_family());
    out
}

// --- Title family (12) -----------------------------------------------------
fn title_family() -> Vec<SlidesTemplate> {
    vec![
        t("title", "Title Slide", SlidesTemplateFamily::Title, vec![title_slot()], "centered", "Type"),
        t("title_subtitle", "Title + Subtitle", SlidesTemplateFamily::Title, vec![title_slot(), subtitle_slot()], "centered", "Type"),
        t("title_image", "Title + Image", SlidesTemplateFamily::Title, vec![title_slot(), image_slot("landscape")], "split_right", "Image"),
        t("title_with_logo", "Title + Logo", SlidesTemplateFamily::Title, vec![title_slot(), subtitle_slot()], "centered_with_logo", "Image"),
        t("title_centered", "Centered Title", SlidesTemplateFamily::Title, vec![title_slot()], "centered", "Type"),
        t("title_left", "Left-Aligned Title", SlidesTemplateFamily::Title, vec![title_slot(), subtitle_slot()], "left", "Type"),
        t("cover_hero", "Hero Cover", SlidesTemplateFamily::Title, vec![title_slot(), image_slot("landscape")], "full_bleed", "Image"),
        t("cover_split", "Split Cover", SlidesTemplateFamily::Title, vec![title_slot(), subtitle_slot(), image_slot("landscape")], "split_left", "Image"),
        t("cover_video", "Video Cover", SlidesTemplateFamily::Title, vec![title_slot(), subtitle_slot()], "video_bg", "Video"),
        t("title_with_author", "Title + Author", SlidesTemplateFamily::Title, vec![title_slot(), SlotDef::new("author", SlotType::PersonName, "Author")], "centered", "User"),
        t("title_with_date", "Title + Date", SlidesTemplateFamily::Title, vec![title_slot(), SlotDef::new("date", SlotType::LabelText, "Date").optional()], "centered", "Calendar"),
        t("title_event", "Event Title", SlidesTemplateFamily::Title, vec![title_slot(), SlotDef::new("event_date", SlotType::LabelText, "Event Date"), SlotDef::new("location", SlotType::LabelText, "Location").optional()], "centered", "Calendar"),
    ]
}

// --- Agenda family (8) -----------------------------------------------------
fn agenda_family() -> Vec<SlidesTemplate> {
    vec![
        t("agenda", "Agenda", SlidesTemplateFamily::Agenda, vec![title_slot(), bullets_slot(2, 8)], "left_bullets", "List"),
        t("agenda_numbered", "Numbered Agenda", SlidesTemplateFamily::Agenda, vec![title_slot(), numbered_slot(2, 8)], "numbered", "ListOrdered"),
        t("agenda_with_timing", "Agenda + Timing", SlidesTemplateFamily::Agenda, vec![title_slot(), SlotDef::new("items", SlotType::NumberedList, "Items").items(2, 8), SlotDef::new("timings", SlotType::NumberedList, "Timings").items(2, 8)], "two_col", "Clock"),
        t("table_of_contents", "Table of Contents", SlidesTemplateFamily::Agenda, vec![title_slot(), numbered_slot(2, 12)], "toc", "List"),
        t("toc_two_col", "TOC Two-Column", SlidesTemplateFamily::Agenda, vec![title_slot(), numbered_slot(2, 6), SlotDef::new("items_right", SlotType::NumberedList, "Right Items").items(2, 6).optional()], "two_col", "List"),
        t("meeting_overview", "Meeting Overview", SlidesTemplateFamily::Agenda, vec![title_slot(), bullets_slot(2, 6), SlotDef::new("duration", SlotType::LabelText, "Duration").optional()], "left_bullets", "Users"),
        t("roadmap_agenda", "Roadmap Agenda", SlidesTemplateFamily::Agenda, vec![title_slot(), steps_slot(2, 6)], "horizontal_steps", "Map"),
        t("workshop_agenda", "Workshop Agenda", SlidesTemplateFamily::Agenda, vec![title_slot(), SlotDef::new("sessions", SlotType::StepList, "Sessions").items(2, 6)], "timeline", "Users"),
        t("agenda_linkable", "Linkable Agenda", SlidesTemplateFamily::Agenda, vec![title_slot(), numbered_slot(2, 8), SlotDef::new("links", SlotType::NumberedList, "Links").items(2, 8).optional()], "linkable", "Link"),
        t("toc_with_page_numbers", "TOC + Page Numbers", SlidesTemplateFamily::Agenda, vec![title_slot(), numbered_slot(2, 12), SlotDef::new("pages", SlotType::NumberedList, "Pages").items(2, 12)], "toc_pages", "BookOpen"),
        t("agenda_two_col", "Two-Column Agenda", SlidesTemplateFamily::Agenda, vec![title_slot(), numbered_slot(2, 6), SlotDef::new("items_right", SlotType::NumberedList, "Right Items").items(2, 6).optional()], "two_col", "Columns2"),
        t("agenda_with_breaks", "Agenda + Breaks", SlidesTemplateFamily::Agenda, vec![title_slot(), SlotDef::new("sessions", SlotType::StepList, "Sessions").items(2, 8), SlotDef::new("breaks", SlotType::DateList, "Breaks").items(1, 4).optional()], "timeline", "Coffee"),
        t("agenda_track", "Multi-Track Agenda", SlidesTemplateFamily::Agenda, vec![title_slot(), SlotDef::new("track_a", SlotType::StepList, "Track A").items(2, 6), SlotDef::new("track_b", SlotType::StepList, "Track B").items(2, 6).optional()], "tracks", "GitBranch"),
        t("agenda_with_speakers", "Agenda + Speakers", SlidesTemplateFamily::Agenda, vec![title_slot(), SlotDef::new("sessions", SlotType::StepList, "Sessions").items(2, 8), SlotDef::new("speakers", SlotType::NumberedList, "Speakers").items(2, 8).optional()], "timeline_speakers", "Users"),
        t("agenda_q_a", "Agenda + Q&A", SlidesTemplateFamily::Agenda, vec![title_slot(), bullets_slot(2, 6), SlotDef::new("qa_time", SlotType::LabelText, "Q&A Time").optional()], "left_bullets", "HelpCircle"),
        t("agenda_with_goals", "Agenda + Goals", SlidesTemplateFamily::Agenda, vec![title_slot(), bullets_slot(2, 6), SlotDef::new("goals", SlotType::BulletList, "Goals").items(1, 4).optional()], "left_bullets_goals", "Target"),
    ]
}

// --- Bullet family (20) ----------------------------------------------------
fn bullet_family() -> Vec<SlidesTemplate> {
    let mut v = vec![
        with_max_bullets(t("bullet", "Bullet List", SlidesTemplateFamily::Bullet, vec![title_slot(), bullets_slot(2, 8)], "left_bullets", "List"), 8),
        with_max_bullets(t("bullet_with_icon", "Bullets + Icons", SlidesTemplateFamily::Bullet, vec![title_slot(), bullets_slot(2, 6)], "icon_bullets", "List"), 6),
        with_max_bullets(t("numbered_list", "Numbered List", SlidesTemplateFamily::Bullet, vec![title_slot(), numbered_slot(2, 8)], "numbered", "ListOrdered"), 8),
        with_max_bullets(t("bullet_two_col", "Two-Column Bullets", SlidesTemplateFamily::Bullet, vec![title_slot(), bullets_slot(2, 6), SlotDef::new("bullets_right", SlotType::BulletList, "Right Bullets").items(2, 6).optional()], "two_col", "Columns2"), 12),
        with_max_bullets(t("bullet_three_col", "Three-Column Bullets", SlidesTemplateFamily::Bullet, vec![title_slot(), bullets_slot(2, 4), SlotDef::new("col2", SlotType::BulletList, "Column 2").items(2, 4).optional(), SlotDef::new("col3", SlotType::BulletList, "Column 3").items(2, 4).optional()], "three_col", "Columns3"), 12),
        t("paragraph", "Paragraph", SlidesTemplateFamily::Bullet, vec![title_slot(), body_slot()], "paragraph", "Text"),
        t("key_takeaway", "Key Takeaway", SlidesTemplateFamily::Bullet, vec![title_slot(), body_slot()], "callout", "Lightbulb"),
        with_image_orientation(t("key_takeaway_with_image", "Takeaway + Image", SlidesTemplateFamily::Bullet, vec![title_slot(), body_slot(), image_slot("landscape")], "split_right", "Lightbulb"), "landscape"),
        with_max_bullets(t("bullet_with_sub_bullets", "Bullets + Sub-bullets", SlidesTemplateFamily::Bullet, vec![title_slot(), bullets_slot(2, 6)], "nested_bullets", "ListTree"), 6),
        with_max_bullets(t("checklist", "Checklist", SlidesTemplateFamily::Bullet, vec![title_slot(), bullets_slot(2, 10)], "checklist", "CheckSquare"), 10),
        t("definition_list", "Definition List", SlidesTemplateFamily::Bullet, vec![title_slot(), SlotDef::new("terms", SlotType::NumberedList, "Terms").items(2, 6), SlotDef::new("definitions", SlotType::NumberedList, "Definitions").items(2, 6)], "two_col", "BookOpen"),
        with_image_orientation(t("bullet_with_image", "Bullets + Image", SlidesTemplateFamily::Bullet, vec![title_slot(), bullets_slot(2, 5), image_slot("landscape")], "split_right", "Image"), "landscape"),
        t("callout_box", "Callout Box", SlidesTemplateFamily::Bullet, vec![title_slot(), body_slot()], "callout_box", "MessageSquare"),
        t("sticky_note", "Sticky Note", SlidesTemplateFamily::Bullet, vec![title_slot(), body_slot()], "sticky", "StickyNote"),
        with_max_bullets(t("bullet_pill", "Pill Bullets", SlidesTemplateFamily::Bullet, vec![title_slot(), bullets_slot(2, 6)], "pill_bullets", "Pill"), 6),
        with_max_bullets(t("bullet_arrow", "Arrow Bullets", SlidesTemplateFamily::Bullet, vec![title_slot(), bullets_slot(2, 6)], "arrow_bullets", "ArrowRight"), 6),
        with_max_bullets(t("bullet_chevron", "Chevron Bullets", SlidesTemplateFamily::Bullet, vec![title_slot(), bullets_slot(2, 6)], "chevron_bullets", "ChevronRight"), 6),
        with_max_bullets(t("bullet_star", "Star Bullets", SlidesTemplateFamily::Bullet, vec![title_slot(), bullets_slot(2, 6)], "star_bullets", "Star"), 6),
        t("text_highlight", "Highlighted Text", SlidesTemplateFamily::Bullet, vec![title_slot(), body_slot()], "highlight", "Highlighter"),
        with_max_bullets(t("bullet_with_stat", "Bullets + Stat", SlidesTemplateFamily::Bullet, vec![title_slot(), bullets_slot(2, 5), stat_number_slot(), stat_label_slot()], "split_right_stat", "BarChart3"), 5),
    ];
    v
}

// --- Quote family (10) -----------------------------------------------------
fn quote_family() -> Vec<SlidesTemplate> {
    vec![
        t("quote", "Quote", SlidesTemplateFamily::Quote, vec![quote_slot(), attribution_slot()], "centered", "Quote"),
        with_image_orientation(t("quote_with_image", "Quote + Image", SlidesTemplateFamily::Quote, vec![quote_slot(), attribution_slot(), image_slot("landscape")], "split_right", "Quote"), "landscape"),
        t("pull_quote", "Pull Quote", SlidesTemplateFamily::Quote, vec![quote_slot(), attribution_slot()], "pull", "Quote"),
        t("quote_large", "Large Quote", SlidesTemplateFamily::Quote, vec![quote_slot(), attribution_slot()], "large_centered", "Quote"),
        t("quote_with_author", "Quote + Author", SlidesTemplateFamily::Quote, vec![quote_slot(), person_name_slot(), person_role_slot()], "centered", "Quote"),
        t("quote_centered", "Centered Quote", SlidesTemplateFamily::Quote, vec![quote_slot(), attribution_slot()], "centered", "Quote"),
        t("quote_left", "Left Quote", SlidesTemplateFamily::Quote, vec![quote_slot(), attribution_slot()], "left", "Quote"),
        t("testimonial_quote", "Testimonial", SlidesTemplateFamily::Quote, vec![quote_slot(), person_name_slot(), person_role_slot()], "testimonial", "MessageCircle"),
        with_image_orientation(t("quote_with_background", "Quote + Background", SlidesTemplateFamily::Quote, vec![quote_slot(), attribution_slot(), image_slot("landscape")], "full_bleed", "Quote"), "landscape"),
        t("quote_block", "Block Quote", SlidesTemplateFamily::Quote, vec![quote_slot(), attribution_slot()], "block", "Quote"),
    ]
}

// --- Comparison family (18) ------------------------------------------------
fn comparison_family() -> Vec<SlidesTemplate> {
    let col = |n: usize| SlotDef::new(&format!("col{}", n), SlotType::BulletList, &format!("Column {}", n)).items(2, 6);
    let col_title = |n: usize| SlotDef::new(&format!("col{}_title", n), SlotType::LabelText, &format!("Column {} Title", n));
    vec![
        t("comparison_two_col", "Two-Column Comparison", SlidesTemplateFamily::Comparison, vec![title_slot(), col_title(1), col(1), col_title(2), col(2)], "two_col", "Columns2"),
        t("comparison_three_col", "Three-Column Comparison", SlidesTemplateFamily::Comparison, vec![title_slot(), col_title(1), col(1), col_title(2), col(2), col_title(3), col(3)], "three_col", "Columns3"),
        t("comparison_four_col", "Four-Column Comparison", SlidesTemplateFamily::Comparison, vec![title_slot(), col_title(1), col(1), col_title(2), col(2), col_title(3), col(3), col_title(4), col(4)], "four_col", "Columns4"),
        t("pros_cons", "Pros & Cons", SlidesTemplateFamily::Comparison, vec![title_slot(), SlotDef::new("pros", SlotType::BulletList, "Pros").items(2, 6), SlotDef::new("cons", SlotType::BulletList, "Cons").items(2, 6)], "two_col", "Scale"),
        t("versus", "Versus", SlidesTemplateFamily::Comparison, vec![title_slot(), SlotDef::new("left_label", SlotType::LabelText, "Left"), SlotDef::new("right_label", SlotType::LabelText, "Right"), SlotDef::new("left_points", SlotType::BulletList, "Left Points").items(2, 5), SlotDef::new("right_points", SlotType::BulletList, "Right Points").items(2, 5)], "versus", "Swords"),
        t("before_after", "Before / After", SlidesTemplateFamily::Comparison, vec![title_slot(), SlotDef::new("before", SlotType::BodyText, "Before"), SlotDef::new("after", SlotType::BodyText, "After")], "two_col", "ArrowLeftRight"),
        with_image_orientation(t("comparison_with_image", "Comparison + Image", SlidesTemplateFamily::Comparison, vec![title_slot(), col_title(1), col(1), col_title(2), col(2), image_slot("square")], "two_col_image", "Image"), "square"),
        t("comparison_table", "Comparison Table", SlidesTemplateFamily::Comparison, vec![title_slot(), SlotDef::new("headers", SlotType::NumberedList, "Headers").items(2, 5), SlotDef::new("rows", SlotType::BulletList, "Rows").items(2, 8)], "table", "Table"),
        t("feature_matrix", "Feature Matrix", SlidesTemplateFamily::Comparison, vec![title_slot(), SlotDef::new("features", SlotType::BulletList, "Features").items(2, 8), SlotDef::new("options", SlotType::NumberedList, "Options").items(2, 4)], "matrix", "Grid3x3"),
        t("pricing_comparison", "Pricing Comparison", SlidesTemplateFamily::Comparison, vec![title_slot(), SlotDef::new("plans", SlotType::NumberedList, "Plans").items(2, 4), SlotDef::new("prices", SlotType::NumberedList, "Prices").items(2, 4), SlotDef::new("features", SlotType::BulletList, "Features").items(2, 8)], "pricing", "DollarSign"),
        t("comparison_split", "Split Comparison", SlidesTemplateFamily::Comparison, vec![title_slot(), SlotDef::new("left", SlotType::BodyText, "Left"), SlotDef::new("right", SlotType::BodyText, "Right")], "split", "Split"),
        t("comparison_overlap", "Overlap Comparison", SlidesTemplateFamily::Comparison, vec![title_slot(), SlotDef::new("left", SlotType::BulletList, "Left").items(2, 5), SlotDef::new("overlap", SlotType::BulletList, "Overlap").items(1, 3), SlotDef::new("right", SlotType::BulletList, "Right").items(2, 5)], "venn", "Circle"),
        t("comparison_stack", "Stacked Comparison", SlidesTemplateFamily::Comparison, vec![title_slot(), col(1), col(2)], "stacked", "Layers"),
        t("comparison_pill", "Pill Comparison", SlidesTemplateFamily::Comparison, vec![title_slot(), col_title(1), col(1), col_title(2), col(2)], "pill", "Pill"),
        t("comparison_arrow", "Arrow Comparison", SlidesTemplateFamily::Comparison, vec![title_slot(), SlotDef::new("from", SlotType::BodyText, "From"), SlotDef::new("to", SlotType::BodyText, "To")], "arrow", "ArrowRight"),
        t("comparison_scale", "Scale Comparison", SlidesTemplateFamily::Comparison, vec![title_slot(), SlotDef::new("low", SlotType::LabelText, "Low"), SlotDef::new("high", SlotType::LabelText, "High"), SlotDef::new("items", SlotType::BulletList, "Items").items(2, 6)], "scale", "Scale"),
        t("comparison_balance", "Balance Comparison", SlidesTemplateFamily::Comparison, vec![title_slot(), SlotDef::new("left_weight", SlotType::StatNumber, "Left Weight"), SlotDef::new("right_weight", SlotType::StatNumber, "Right Weight"), SlotDef::new("left_label", SlotType::LabelText, "Left"), SlotDef::new("right_label", SlotType::LabelText, "Right")], "balance", "Scale"),
        t("comparison_slider", "Slider Comparison", SlidesTemplateFamily::Comparison, vec![title_slot(), SlotDef::new("left", SlotType::BodyText, "Left"), SlotDef::new("right", SlotType::BodyText, "Right"), SlotDef::new("position", SlotType::StatNumber, "Position")], "slider", "SlidersHorizontal"),
    ]
}

// --- Timeline family (16) --------------------------------------------------
fn timeline_family() -> Vec<SlidesTemplate> {
    vec![
        t("timeline_horizontal", "Horizontal Timeline", SlidesTemplateFamily::Timeline, vec![title_slot(), steps_slot(3, 8), dates_slot(3, 8)], "horizontal_timeline", "Clock"),
        t("timeline_vertical", "Vertical Timeline", SlidesTemplateFamily::Timeline, vec![title_slot(), steps_slot(3, 8), dates_slot(3, 8)], "vertical_timeline", "Clock"),
        t("timeline_circular", "Circular Timeline", SlidesTemplateFamily::Timeline, vec![title_slot(), steps_slot(3, 8)], "circular", "Circle"),
        t("timeline_zigzag", "Zigzag Timeline", SlidesTemplateFamily::Timeline, vec![title_slot(), steps_slot(3, 8), dates_slot(3, 8)], "zigzag", "Wave"),
        t("milestone", "Milestone", SlidesTemplateFamily::Timeline, vec![title_slot(), SlotDef::new("milestone", SlotType::LabelText, "Milestone"), SlotDef::new("date", SlotType::LabelText, "Date"), body_slot()], "milestone", "Flag"),
        with_image_orientation(t("milestone_with_image", "Milestone + Image", SlidesTemplateFamily::Timeline, vec![title_slot(), SlotDef::new("milestone", SlotType::LabelText, "Milestone"), body_slot(), image_slot("landscape")], "milestone_image", "Flag"), "landscape"),
        t("roadmap", "Roadmap", SlidesTemplateFamily::Timeline, vec![title_slot(), steps_slot(3, 6), dates_slot(3, 6)], "roadmap", "Map"),
        t("roadmap_with_phases", "Roadmap + Phases", SlidesTemplateFamily::Timeline, vec![title_slot(), SlotDef::new("phases", SlotType::StepList, "Phases").items(2, 5), SlotDef::new("phase_dates", SlotType::DateList, "Phase Dates").items(2, 5)], "phases", "Map"),
        t("gantt_summary", "Gantt Summary", SlidesTemplateFamily::Timeline, vec![title_slot(), SlotDef::new("tasks", SlotType::StepList, "Tasks").items(2, 8), SlotDef::new("durations", SlotType::NumberedList, "Durations").items(2, 8)], "gantt", "BarChart3"),
        t("process_timeline", "Process Timeline", SlidesTemplateFamily::Timeline, vec![title_slot(), steps_slot(3, 8)], "process", "GitBranch"),
        t("event_timeline", "Event Timeline", SlidesTemplateFamily::Timeline, vec![title_slot(), SlotDef::new("events", SlotType::StepList, "Events").items(3, 8), dates_slot(3, 8)], "events", "Calendar"),
        t("history_timeline", "History Timeline", SlidesTemplateFamily::Timeline, vec![title_slot(), SlotDef::new("events", SlotType::StepList, "Events").items(3, 10), dates_slot(3, 10)], "history", "History"),
        t("timeline_with_dates", "Timeline + Dates", SlidesTemplateFamily::Timeline, vec![title_slot(), steps_slot(3, 8), dates_slot(3, 8)], "dated_timeline", "Calendar"),
        t("timeline_with_icons", "Timeline + Icons", SlidesTemplateFamily::Timeline, vec![title_slot(), steps_slot(3, 8)], "icon_timeline", "Sparkles"),
        t("timeline_reverse", "Reverse Timeline", SlidesTemplateFamily::Timeline, vec![title_slot(), steps_slot(3, 8), dates_slot(3, 8)], "reverse_timeline", "Rewind"),
        t("timeline_convergent", "Convergent Timeline", SlidesTemplateFamily::Timeline, vec![title_slot(), SlotDef::new("streams", SlotType::StepList, "Streams").items(2, 4), SlotDef::new("convergence", SlotType::LabelText, "Convergence")], "convergent", "Merge"),
    ]
}

// --- Image family (24) -----------------------------------------------------
fn image_family() -> Vec<SlidesTemplate> {
    let img = |id: &str, label: &str, layout: &str, o: &'static str, extra: Vec<SlotDef>| {
        let mut slots = vec![title_slot(), image_slot(o)];
        slots.extend(extra);
        with_image_orientation(t(id, label, SlidesTemplateFamily::Image, slots, layout, "Image"), o)
    };
    vec![
        img("image_full_bleed", "Full-Bleed Image", "full_bleed", "landscape", vec![caption_slot()]),
        img("image_grid_2x2", "Image Grid 2×2", "grid_2x2", "square", vec![SlotDef::new("images_extra", SlotType::ImageQuery, "Extra Images").items(3, 3)]),
        img("image_grid_3x3", "Image Grid 3×3", "grid_3x3", "square", vec![SlotDef::new("images_extra", SlotType::ImageQuery, "Extra Images").items(8, 8)]),
        img("image_grid_4x4", "Image Grid 4×4", "grid_4x4", "square", vec![SlotDef::new("images_extra", SlotType::ImageQuery, "Extra Images").items(15, 15)]),
        img("image_with_caption", "Image + Caption", "centered", "landscape", vec![caption_slot()]),
        img("image_left_text_right", "Image Left + Text Right", "split_left", "landscape", vec![body_slot()]),
        img("image_right_text_left", "Text Left + Image Right", "split_right", "landscape", vec![body_slot()]),
        img("image_top_text_bottom", "Image Top + Text Bottom", "stack_top", "landscape", vec![body_slot()]),
        img("image_bottom_text_top", "Text Top + Image Bottom", "stack_bottom", "landscape", vec![body_slot()]),
        img("photo_collage", "Photo Collage", "collage", "square", vec![SlotDef::new("images_extra", SlotType::ImageQuery, "Extra Images").items(2, 5)]),
        img("hero_image", "Hero Image", "hero", "landscape", vec![subtitle_slot()]),
        img("image_stack", "Image Stack", "stack", "landscape", vec![SlotDef::new("images_extra", SlotType::ImageQuery, "Extra Images").items(1, 3)]),
        img("image_with_overlay", "Image + Overlay", "overlay", "landscape", vec![body_slot()]),
        img("image_with_quote", "Image + Quote", "overlay_quote", "landscape", vec![quote_slot(), attribution_slot()]),
        img("image_with_stat", "Image + Stat", "overlay_stat", "landscape", vec![stat_number_slot(), stat_label_slot()]),
        img("image_pair", "Image Pair", "pair", "landscape", vec![SlotDef::new("image_right", SlotType::ImageQuery, "Right Image")]),
        img("image_triple", "Image Triple", "triple", "landscape", vec![SlotDef::new("image_middle", SlotType::ImageQuery, "Middle Image"), SlotDef::new("image_right", SlotType::ImageQuery, "Right Image")]),
        img("image_with_bullet", "Image + Bullets", "split_right_bullets", "landscape", vec![bullets_slot(2, 5)]),
        img("image_centered", "Centered Image", "centered", "square", vec![caption_slot()]),
        img("image_with_attribution", "Image + Attribution", "centered_attr", "landscape", vec![caption_slot(), attribution_slot()]),
        img("image_with_logo", "Image + Logo", "centered_logo", "landscape", vec![]),
        img("image_background_text", "Background Image + Text", "bg_text", "landscape", vec![body_slot()]),
        img("image_gallery", "Image Gallery", "gallery", "square", vec![SlotDef::new("images_extra", SlotType::ImageQuery, "Extra Images").items(3, 7)]),
        img("image_carousel", "Image Carousel", "carousel", "landscape", vec![SlotDef::new("images_extra", SlotType::ImageQuery, "Extra Images").items(2, 6)]),
    ]
}

// --- Chart family (22) -----------------------------------------------------
fn chart_family() -> Vec<SlidesTemplate> {
    let chart = |id: &str, label: &str, layout: &str, extra: Vec<SlotDef>| {
        let mut slots = vec![title_slot(), chart_series_slot()];
        slots.extend(extra);
        with_chart(t(id, label, SlidesTemplateFamily::Chart, slots, layout, "BarChart3"))
    };
    vec![
        chart("bar_chart", "Bar Chart", "bar", vec![]),
        chart("bar_chart_grouped", "Grouped Bar Chart", "bar_grouped", vec![SlotDef::new("series2", SlotType::ChartSeries, "Series 2").optional()]),
        chart("bar_chart_stacked", "Stacked Bar Chart", "bar_stacked", vec![SlotDef::new("series2", SlotType::ChartSeries, "Series 2").optional()]),
        chart("line_chart", "Line Chart", "line", vec![]),
        chart("line_chart_multi", "Multi-Line Chart", "line_multi", vec![SlotDef::new("series2", SlotType::ChartSeries, "Series 2").optional(), SlotDef::new("series3", SlotType::ChartSeries, "Series 3").optional()]),
        chart("pie_chart", "Pie Chart", "pie", vec![]),
        chart("donut_chart", "Donut Chart", "donut", vec![]),
        t("stat_big_number", "Big Number", SlidesTemplateFamily::Chart, vec![title_slot(), stat_number_slot(), stat_label_slot()], "big_number", "Hash"),
        t("stat_big_number_with_label", "Big Number + Label", SlidesTemplateFamily::Chart, vec![title_slot(), stat_number_slot(), stat_label_slot(), SlotDef::new("context", SlotType::BodyText, "Context").optional()], "big_number_context", "Hash"),
        t("kpi_dashboard", "KPI Dashboard", SlidesTemplateFamily::Chart, vec![title_slot(), SlotDef::new("kpis", SlotType::ChartSeries, "KPIs")], "kpi_dashboard", "LayoutDashboard"),
        t("kpi_grid_2", "KPI Grid 2", SlidesTemplateFamily::Chart, vec![title_slot(), SlotDef::new("kpi1_value", SlotType::StatNumber, "KPI 1"), SlotDef::new("kpi1_label", SlotType::StatLabel, "KPI 1 Label"), SlotDef::new("kpi2_value", SlotType::StatNumber, "KPI 2"), SlotDef::new("kpi2_label", SlotType::StatLabel, "KPI 2 Label")], "kpi_grid_2", "Grid2x2"),
        t("kpi_grid_3", "KPI Grid 3", SlidesTemplateFamily::Chart, vec![title_slot(), SlotDef::new("kpi1_value", SlotType::StatNumber, "KPI 1"), SlotDef::new("kpi1_label", SlotType::StatLabel, "KPI 1 Label"), SlotDef::new("kpi2_value", SlotType::StatNumber, "KPI 2"), SlotDef::new("kpi2_label", SlotType::StatLabel, "KPI 2 Label"), SlotDef::new("kpi3_value", SlotType::StatNumber, "KPI 3"), SlotDef::new("kpi3_label", SlotType::StatLabel, "KPI 3 Label")], "kpi_grid_3", "Grid3x3"),
        t("kpi_grid_4", "KPI Grid 4", SlidesTemplateFamily::Chart, vec![title_slot(), SlotDef::new("kpi1_value", SlotType::StatNumber, "KPI 1"), SlotDef::new("kpi1_label", SlotType::StatLabel, "KPI 1 Label"), SlotDef::new("kpi2_value", SlotType::StatNumber, "KPI 2"), SlotDef::new("kpi2_label", SlotType::StatLabel, "KPI 2 Label"), SlotDef::new("kpi3_value", SlotType::StatNumber, "KPI 3"), SlotDef::new("kpi3_label", SlotType::StatLabel, "KPI 3 Label"), SlotDef::new("kpi4_value", SlotType::StatNumber, "KPI 4"), SlotDef::new("kpi4_label", SlotType::StatLabel, "KPI 4 Label")], "kpi_grid_4", "Grid4x4"),
        t("data_table", "Data Table", SlidesTemplateFamily::Chart, vec![title_slot(), SlotDef::new("headers", SlotType::NumberedList, "Headers").items(2, 6), SlotDef::new("rows", SlotType::BulletList, "Rows").items(2, 10)], "table", "Table"),
        t("progress_bar", "Progress Bar", SlidesTemplateFamily::Chart, vec![title_slot(), stat_number_slot(), stat_label_slot()], "progress_bar", "TrendingUp"),
        t("progress_circle", "Progress Circle", SlidesTemplateFamily::Chart, vec![title_slot(), stat_number_slot(), stat_label_slot()], "progress_circle", "Circle"),
        chart("funnel", "Funnel", "funnel", vec![]),
        chart("funnel_with_labels", "Funnel + Labels", "funnel_labels", vec![SlotDef::new("labels", SlotType::NumberedList, "Labels").items(2, 8)]),
        t("gauge", "Gauge", SlidesTemplateFamily::Chart, vec![title_slot(), stat_number_slot(), SlotDef::new("max", SlotType::StatNumber, "Max"), stat_label_slot()], "gauge", "Gauge"),
        chart("area_chart", "Area Chart", "area", vec![]),
        chart("scatter_plot", "Scatter Plot", "scatter", vec![SlotDef::new("series2", SlotType::ChartSeries, "Series 2").optional()]),
        chart("histogram", "Histogram", "histogram", vec![]),
    ]
}

// --- Diagram family (24) ---------------------------------------------------
fn diagram_family() -> Vec<SlidesTemplate> {
    let d = |id: &str, label: &str, layout: &str, slots: Vec<SlotDef>| {
        t(id, label, SlidesTemplateFamily::Diagram, slots, layout, "Workflow")
    };
    vec![
        d("flowchart", "Flowchart", "flowchart", vec![title_slot(), steps_slot(3, 8)]),
        d("flowchart_vertical", "Vertical Flowchart", "flowchart_v", vec![title_slot(), steps_slot(3, 8)]),
        d("process_steps", "Process Steps", "process", vec![title_slot(), steps_slot(3, 8)]),
        d("process_steps_circular", "Circular Process", "process_circular", vec![title_slot(), steps_slot(3, 6)]),
        d("pyramid", "Pyramid", "pyramid", vec![title_slot(), steps_slot(3, 5)]),
        d("pyramid_inverted", "Inverted Pyramid", "pyramid_inv", vec![title_slot(), steps_slot(3, 5)]),
        d("venn_diagram", "Venn Diagram", "venn", vec![title_slot(), SlotDef::new("set_a", SlotType::LabelText, "Set A"), SlotDef::new("set_b", SlotType::LabelText, "Set B"), SlotDef::new("intersection", SlotType::LabelText, "Intersection").optional()]),
        d("venn_three", "Three-Circle Venn", "venn3", vec![title_slot(), SlotDef::new("set_a", SlotType::LabelText, "Set A"), SlotDef::new("set_b", SlotType::LabelText, "Set B"), SlotDef::new("set_c", SlotType::LabelText, "Set C"), SlotDef::new("intersection", SlotType::LabelText, "Intersection").optional()]),
        d("swot", "SWOT Analysis", "swot", vec![title_slot(), SlotDef::new("strengths", SlotType::BulletList, "Strengths").items(2, 5), SlotDef::new("weaknesses", SlotType::BulletList, "Weaknesses").items(2, 5), SlotDef::new("opportunities", SlotType::BulletList, "Opportunities").items(2, 5), SlotDef::new("threats", SlotType::BulletList, "Threats").items(2, 5)]),
        d("puzzle", "Puzzle", "puzzle", vec![title_slot(), steps_slot(2, 6)]),
        d("puzzle_4", "4-Piece Puzzle", "puzzle4", vec![title_slot(), steps_slot(4, 4)]),
        d("hexagon", "Hexagon", "hexagon", vec![title_slot(), steps_slot(3, 6)]),
        d("hexagon_grid", "Hexagon Grid", "hexagon_grid", vec![title_slot(), steps_slot(3, 7)]),
        d("diamond", "Diamond", "diamond", vec![title_slot(), steps_slot(3, 5)]),
        d("circle_segments", "Circle Segments", "circle_segments", vec![title_slot(), steps_slot(3, 8)]),
        d("loop_cycle", "Loop Cycle", "loop", vec![title_slot(), steps_slot(3, 6)]),
        d("cycle_3", "3-Step Cycle", "cycle3", vec![title_slot(), steps_slot(3, 3)]),
        d("cycle_4", "4-Step Cycle", "cycle4", vec![title_slot(), steps_slot(4, 4)]),
        d("cycle_5", "5-Step Cycle", "cycle5", vec![title_slot(), steps_slot(5, 5)]),
        d("matrix_2x2", "2×2 Matrix", "matrix2x2", vec![title_slot(), SlotDef::new("quadrants", SlotType::StepList, "Quadrants").items(4, 4)]),
        d("ladder", "Ladder", "ladder", vec![title_slot(), steps_slot(3, 6)]),
        d("iceberg", "Iceberg", "iceberg", vec![title_slot(), SlotDef::new("above", SlotType::BodyText, "Above Surface"), SlotDef::new("below", SlotType::BodyText, "Below Surface")]),
        d("iceberg_with_layers", "Iceberg + Layers", "iceberg_layers", vec![title_slot(), SlotDef::new("above", SlotType::BulletList, "Above").items(1, 3), SlotDef::new("below", SlotType::BulletList, "Below").items(2, 5)]),
        d("ecosystem", "Ecosystem", "ecosystem", vec![title_slot(), SlotDef::new("components", SlotType::StepList, "Components").items(3, 8), SlotDef::new("connections", SlotType::BodyText, "Connections").optional()]),
    ]
}

// --- Team family (14) ------------------------------------------------------
fn team_family() -> Vec<SlidesTemplate> {
    vec![
        t("team_grid", "Team Grid", SlidesTemplateFamily::Team, vec![title_slot(), SlotDef::new("members", SlotType::StepList, "Members").items(2, 8)], "team_grid", "Users"),
        t("team_grid_2x2", "Team Grid 2×2", SlidesTemplateFamily::Team, vec![title_slot(), SlotDef::new("members", SlotType::StepList, "Members").items(4, 4)], "grid_2x2", "Users"),
        t("team_grid_3x3", "Team Grid 3×3", SlidesTemplateFamily::Team, vec![title_slot(), SlotDef::new("members", SlotType::StepList, "Members").items(9, 9)], "grid_3x3", "Users"),
        t("org_chart", "Org Chart", SlidesTemplateFamily::Team, vec![title_slot(), person_name_slot(), SlotDef::new("reports", SlotType::StepList, "Reports").items(2, 8)], "org_chart", "Network"),
        t("org_chart_two_level", "Org Chart (Two-Level)", SlidesTemplateFamily::Team, vec![title_slot(), person_name_slot(), SlotDef::new("managers", SlotType::StepList, "Managers").items(2, 5), SlotDef::new("reports", SlotType::StepList, "Reports").items(2, 10)], "org_two_level", "Network"),
        with_image_orientation(t("person_card", "Person Card", SlidesTemplateFamily::Team, vec![person_name_slot(), person_role_slot(), image_slot("square")], "person_card", "User"), "square"),
        with_image_orientation(t("person_card_with_image", "Person Card + Image", SlidesTemplateFamily::Team, vec![person_name_slot(), person_role_slot(), body_slot(), image_slot("square")], "person_card_image", "User"), "square"),
        t("testimonial", "Testimonial", SlidesTemplateFamily::Team, vec![quote_slot(), person_name_slot(), person_role_slot()], "testimonial", "MessageCircle"),
        with_image_orientation(t("testimonial_with_image", "Testimonial + Image", SlidesTemplateFamily::Team, vec![quote_slot(), person_name_slot(), person_role_slot(), image_slot("square")], "testimonial_image", "MessageCircle"), "square"),
        t("avatar_list", "Avatar List", SlidesTemplateFamily::Team, vec![title_slot(), SlotDef::new("names", SlotType::NumberedList, "Names").items(2, 8)], "avatar_list", "Users"),
        t("speaker_bio", "Speaker Bio", SlidesTemplateFamily::Team, vec![person_name_slot(), person_role_slot(), body_slot()], "speaker_bio", "Mic"),
        with_image_orientation(t("speaker_bio_with_image", "Speaker Bio + Image", SlidesTemplateFamily::Team, vec![person_name_slot(), person_role_slot(), body_slot(), image_slot("square")], "speaker_bio_image", "Mic"), "square"),
        t("team_with_roles", "Team + Roles", SlidesTemplateFamily::Team, vec![title_slot(), SlotDef::new("names", SlotType::NumberedList, "Names").items(2, 8), SlotDef::new("roles", SlotType::NumberedList, "Roles").items(2, 8)], "team_roles", "Users"),
        t("leadership_team", "Leadership Team", SlidesTemplateFamily::Team, vec![title_slot(), SlotDef::new("leaders", SlotType::StepList, "Leaders").items(2, 6)], "leadership", "Crown"),
    ]
}

// --- Media family (12) -----------------------------------------------------
fn media_family() -> Vec<SlidesTemplate> {
    vec![
        t("video_embed", "Video Embed", SlidesTemplateFamily::Media, vec![title_slot(), SlotDef::new("video_url", SlotType::LabelText, "Video URL"), caption_slot()], "video", "Video"),
        t("icon_list", "Icon List", SlidesTemplateFamily::Media, vec![title_slot(), bullets_slot(2, 8)], "icon_list", "Star"),
        t("icon_grid", "Icon Grid", SlidesTemplateFamily::Media, vec![title_slot(), bullets_slot(4, 12)], "icon_grid", "Grid3x3"),
        t("word_cloud", "Word Cloud", SlidesTemplateFamily::Media, vec![title_slot(), bullets_slot(5, 20)], "word_cloud", "Type"),
        t("word_cloud_weighted", "Weighted Word Cloud", SlidesTemplateFamily::Media, vec![title_slot(), SlotDef::new("words", SlotType::NumberedList, "Words").items(5, 20), SlotDef::new("weights", SlotType::ChartSeries, "Weights")], "word_cloud_weighted", "Type"),
        t("map", "Map", SlidesTemplateFamily::Media, vec![title_slot(), SlotDef::new("locations", SlotType::NumberedList, "Locations").items(1, 8)], "map", "Map"),
        t("map_with_pins", "Map + Pins", SlidesTemplateFamily::Media, vec![title_slot(), SlotDef::new("pins", SlotType::StepList, "Pins").items(1, 8)], "map_pins", "MapPin"),
        t("qr_code", "QR Code", SlidesTemplateFamily::Media, vec![title_slot(), SlotDef::new("url", SlotType::LabelText, "URL"), caption_slot()], "qr", "QrCode"),
        t("audio_player", "Audio Player", SlidesTemplateFamily::Media, vec![title_slot(), SlotDef::new("audio_url", SlotType::LabelText, "Audio URL"), caption_slot()], "audio", "Volume2"),
        t("social_feed", "Social Feed", SlidesTemplateFamily::Media, vec![title_slot(), SlotDef::new("posts", SlotType::StepList, "Posts").items(2, 5)], "social", "MessageSquare"),
        t("screenshot_with_annotation", "Screenshot + Annotation", SlidesTemplateFamily::Media, vec![title_slot(), image_slot("landscape"), SlotDef::new("annotations", SlotType::BulletList, "Annotations").items(1, 5).optional()], "screenshot", "Monitor"),
        t("code_block", "Code Block", SlidesTemplateFamily::Media, vec![title_slot(), SlotDef::new("code", SlotType::BodyText, "Code"), SlotDef::new("language", SlotType::LabelText, "Language").optional()], "code", "Code"),
    ]
}

// --- Section family (12) ---------------------------------------------------
fn section_family() -> Vec<SlidesTemplate> {
    vec![
        t("section_break", "Section Break", SlidesTemplateFamily::Section, vec![section_label_slot()], "section_break", "Separator"),
        t("section_number", "Section Number", SlidesTemplateFamily::Section, vec![SlotDef::new("number", SlotType::StatNumber, "Number"), section_label_slot()], "section_number", "Hash"),
        t("section_icon", "Section Icon", SlidesTemplateFamily::Section, vec![section_label_slot(), SlotDef::new("icon", SlotType::LabelText, "Icon").optional()], "section_icon", "Star"),
        t("divider_quote", "Divider Quote", SlidesTemplateFamily::Section, vec![quote_slot(), attribution_slot()], "divider_quote", "Quote"),
        with_image_orientation(t("divider_full_bleed", "Full-Bleed Divider", SlidesTemplateFamily::Section, vec![section_label_slot(), image_slot("landscape")], "divider_full_bleed", "Image"), "landscape"),
        t("divider_gradient", "Gradient Divider", SlidesTemplateFamily::Section, vec![section_label_slot()], "divider_gradient", "Palette"),
        t("divider_with_logo", "Divider + Logo", SlidesTemplateFamily::Section, vec![section_label_slot()], "divider_logo", "Image"),
        t("transition", "Transition", SlidesTemplateFamily::Section, vec![section_label_slot(), subtitle_slot()], "transition", "ArrowRight"),
        t("recap", "Recap", SlidesTemplateFamily::Section, vec![title_slot(), bullets_slot(2, 6)], "recap", "RotateCcw"),
        t("next_steps", "Next Steps", SlidesTemplateFamily::Section, vec![title_slot(), bullets_slot(2, 6)], "next_steps", "ArrowRight"),
        t("closing", "Closing", SlidesTemplateFamily::Section, vec![title_slot(), body_slot()], "closing", "Flag"),
        t("thank_you", "Thank You", SlidesTemplateFamily::Section, vec![title_slot(), subtitle_slot()], "thank_you", "Heart"),
        t("qa_section", "Q&A Section", SlidesTemplateFamily::Section, vec![section_label_slot(), SlotDef::new("questions", SlotType::BulletList, "Questions").items(1, 5).optional()], "qa", "HelpCircle"),
        t("resources", "Resources", SlidesTemplateFamily::Section, vec![title_slot(), SlotDef::new("resources", SlotType::BulletList, "Resources").items(2, 8)], "resources", "BookOpen"),
        t("appendix", "Appendix", SlidesTemplateFamily::Section, vec![section_label_slot(), bullets_slot(2, 8)], "appendix", "Paperclip"),
        t("glossary", "Glossary", SlidesTemplateFamily::Section, vec![title_slot(), SlotDef::new("terms", SlotType::NumberedList, "Terms").items(2, 8), SlotDef::new("definitions", SlotType::NumberedList, "Definitions").items(2, 8)], "two_col", "BookOpen"),
        t("references", "References", SlidesTemplateFamily::Section, vec![title_slot(), SlotDef::new("references", SlotType::NumberedList, "References").items(2, 10)], "references", "BookMarked"),
        t("contact", "Contact Slide", SlidesTemplateFamily::Section, vec![title_slot(), person_name_slot(), SlotDef::new("email", SlotType::LabelText, "Email").optional(), SlotDef::new("phone", SlotType::LabelText, "Phone").optional()], "contact", "Mail"),
        t("credits", "Credits", SlidesTemplateFamily::Section, vec![title_slot(), SlotDef::new("credits", SlotType::BulletList, "Credits").items(2, 8)], "credits", "Award"),
        t("summary", "Summary", SlidesTemplateFamily::Section, vec![title_slot(), bullets_slot(2, 8)], "summary", "FileText"),
        t("call_to_action", "Call to Action", SlidesTemplateFamily::Section, vec![title_slot(), body_slot(), SlotDef::new("cta_label", SlotType::LabelText, "CTA Label").optional()], "cta", "MousePointerClick"),
        t("discussion", "Discussion", SlidesTemplateFamily::Section, vec![title_slot(), SlotDef::new("prompts", SlotType::BulletList, "Discussion Prompts").items(2, 6)], "discussion", "MessageCircle"),
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_has_210_templates() {
        let reg = SlidesTemplateRegistry::new();
        assert_eq!(reg.len(), 210, "expected 210 smart templates");
    }

    #[test]
    fn test_no_duplicate_ids() {
        let reg = SlidesTemplateRegistry::new();
        let ids: Vec<&str> = reg.all().iter().map(|t| t.id.as_str()).collect();
        let unique: std::collections::HashSet<&str> = ids.iter().copied().collect();
        assert_eq!(ids.len(), unique.len(), "duplicate template IDs");
    }

    #[test]
    fn test_family_counts() {
        let reg = SlidesTemplateRegistry::new();
        assert_eq!(reg.by_family(SlidesTemplateFamily::Title).len(), 12);
        assert_eq!(reg.by_family(SlidesTemplateFamily::Agenda).len(), 16);
        assert_eq!(reg.by_family(SlidesTemplateFamily::Bullet).len(), 20);
        assert_eq!(reg.by_family(SlidesTemplateFamily::Quote).len(), 10);
        assert_eq!(reg.by_family(SlidesTemplateFamily::Comparison).len(), 18);
        assert_eq!(reg.by_family(SlidesTemplateFamily::Timeline).len(), 16);
        assert_eq!(reg.by_family(SlidesTemplateFamily::Image).len(), 24);
        assert_eq!(reg.by_family(SlidesTemplateFamily::Chart).len(), 22);
        assert_eq!(reg.by_family(SlidesTemplateFamily::Diagram).len(), 24);
        assert_eq!(reg.by_family(SlidesTemplateFamily::Team).len(), 14);
        assert_eq!(reg.by_family(SlidesTemplateFamily::Media).len(), 12);
        assert_eq!(reg.by_family(SlidesTemplateFamily::Section).len(), 22);
    }

    #[test]
    fn test_get_template_by_id() {
        let reg = SlidesTemplateRegistry::new();
        let tmpl = reg.get("title").unwrap();
        assert_eq!(tmpl.label, "Title Slide");
        assert_eq!(tmpl.family, SlidesTemplateFamily::Title);
    }

    #[test]
    fn test_get_nonexistent_template() {
        let reg = SlidesTemplateRegistry::new();
        assert!(reg.get("nonexistent").is_none());
    }

    #[test]
    fn test_every_template_has_valid_slot_schema() {
        let reg = SlidesTemplateRegistry::new();
        for tmpl in reg.all() {
            let schema = tmpl.slot_schema();
            assert!(schema.is_object(), "template {} schema not object", tmpl.id);
            assert!(schema.get("properties").is_some(), "template {} missing properties", tmpl.id);
        }
    }

    #[test]
    fn test_every_template_has_title_or_section_label() {
        let reg = SlidesTemplateRegistry::new();
        for tmpl in reg.all() {
            let has_title = tmpl.slots.iter().any(|s| s.id == "title" || s.id == "section_label" || s.id == "name" || s.id == "quote");
            assert!(has_title, "template {} has no title/section_label/name/quote slot", tmpl.id);
        }
    }

    #[test]
    fn test_chart_templates_support_chart() {
        let reg = SlidesTemplateRegistry::new();
        for tmpl in reg.by_family(SlidesTemplateFamily::Chart) {
            if tmpl.id.contains("chart") || tmpl.id.contains("funnel") || tmpl.id.contains("histogram") || tmpl.id.contains("scatter") || tmpl.id.contains("area") {
                assert!(tmpl.supports_chart, "chart template {} should support_chart", tmpl.id);
            }
        }
    }

    #[test]
    fn test_compact_catalog() {
        let reg = SlidesTemplateRegistry::new();
        let catalog = reg.compact_catalog();
        assert!(catalog.contains("title|Title Slide|Title"));
        assert!(catalog.lines().count() == 210);
    }

    #[test]
    fn test_image_templates_have_orientation_hint() {
        let reg = SlidesTemplateRegistry::new();
        for tmpl in reg.by_family(SlidesTemplateFamily::Image) {
            assert!(tmpl.image_orientation_hint.is_some(), "image template {} should have orientation hint", tmpl.id);
        }
    }

    #[test]
    fn test_slot_schema_required_fields() {
        let reg = SlidesTemplateRegistry::new();
        let tmpl = reg.get("title").unwrap();
        let schema = tmpl.slot_schema();
        let required = schema.get("required").unwrap().as_array().unwrap();
        assert!(required.iter().any(|v| v.as_str() == Some("title")));
        // title_subtitle has subtitle as optional, so it should NOT be in required
        let tmpl2 = reg.get("title_subtitle").unwrap();
        let schema2 = tmpl2.slot_schema();
        let required2 = schema2.get("required").unwrap().as_array().unwrap();
        assert!(required2.iter().any(|v| v.as_str() == Some("title")));
        assert!(!required2.iter().any(|v| v.as_str() == Some("subtitle")));
    }
}
