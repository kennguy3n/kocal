//! Document AI skills — declarative skill definitions for the generative plane.
//!
//! Each skill is a `SkillDef` struct that declares its scope, mode, token budget,
//! prompt builder, LoRA task, and grammar constraint. Skills are data, not code:
//! the prompt builder, grammar selector, and LoRA resolver all operate on the
//! struct's fields.
//!
//! Skills are organized into three surfaces:
//! - **Read**: analyze/extract info from a document (no modification)
//! - **Edit**: refine/transform existing text
//! - **Create**: generate new content from a brief

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Which document surface a skill operates on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SkillSurface {
    /// Read surface — analysis & extraction, no document modification.
    Read,
    /// Edit surface — refine/transform existing text.
    Edit,
    /// Create surface — generate new content from a brief.
    Create,
}

/// What part of the document a skill operates on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SkillScope {
    /// Operates at the cursor position (needs ±200 chars + nearest heading).
    Cursor,
    /// Operates on highlighted text (needs selection + local context).
    Selection,
    /// Operates on the current section (heading → next heading).
    Section,
    /// Operates on the entire document (chunked if large).
    Document,
    /// Generates from a user-provided topic/brief.
    Topic,
}

/// How a skill is executed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SkillMode {
    /// No additional input — executes immediately.
    OneClick,
    /// Needs a free-form text instruction from the user.
    PromptInput,
    /// Needs structured form fields (dropdowns, text fields).
    FormInput,
    /// Breaks work into multiple LLM calls (chunk → process → stitch).
    MultiStep,
}

/// Logical grouping for UI section headers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SkillGroup {
    Refine,
    Extract,
    Generate,
    Document,
}

/// Minimum device tier required for a skill.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SkillTier {
    /// Works on all devices including Low tier.
    Low,
    /// Requires at least Medium tier.
    Medium,
    /// Requires High tier (large context windows).
    High,
}

impl SkillTier {
    /// Maximum output tokens per tier (min, max).
    /// Mirrors `DeviceTier::output_cap()` from kchat-core.
    pub fn output_cap(self) -> (usize, usize) {
        match self {
            SkillTier::Low => (64, 192),
            SkillTier::Medium => (256, 512),
            SkillTier::High => (512, 1024),
        }
    }

    /// Context window cap in tokens (desktop defaults).
    /// Mirrors `DeviceTier::context_cap_for_platform("macos")`.
    pub fn context_cap(self) -> usize {
        match self {
            SkillTier::Low => 2048,
            SkillTier::Medium => 4096,
            SkillTier::High => 16384,
        }
    }

    /// Skills that benefit from thinking mode (reasoning-heavy tasks).
    /// All other skills use `/no_think` regardless of tier.
    pub fn allows_thinking_for(skill_id: &str) -> bool {
        matches!(skill_id, "doc_find_contradictions")
    }
}

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------

/// A sub-variant shown as a flyout submenu (e.g. tone options, language picker).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillSubVariant {
    pub id: String,
    pub label: String,
    /// Context string injected into the system prompt.
    pub context: String,
}

/// A declarative skill definition.
///
/// Skills are data: the generation pipeline reads these fields to build prompts,
/// select LoRA adapters, enforce grammar constraints, and manage token budgets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillDef {
    /// Unique skill identifier (e.g. "edit_fix_grammar").
    pub id: String,
    /// Human-readable label (e.g. "Fix Grammar").
    pub label: String,
    /// Short description shown in UI.
    pub description: String,
    /// Icon name (matches lucide-react icon names).
    pub icon: String,
    /// Which document surface this skill belongs to.
    pub surface: SkillSurface,
    /// Logical group for UI section headers.
    pub group: SkillGroup,
    /// What part of the document this skill operates on.
    pub scope: SkillScope,
    /// How this skill is executed.
    pub mode: SkillMode,
    /// Maximum output tokens.
    pub max_tokens: usize,
    /// Sampling temperature (0.0–1.0).
    pub temperature: f32,
    /// Stop sequences.
    pub stop: Vec<String>,
    /// Text prepended to the model's response to anchor output format.
    pub response_prefix: Option<String>,
    /// Sub-variants shown as a flyout submenu.
    pub sub_variants: Vec<SkillSubVariant>,
    /// Needs a topic/brief input from the user.
    pub needs_topic: bool,
    /// Supports optional keywords input.
    pub supports_keywords: bool,
    /// Custom label for the topic input field.
    pub topic_label: Option<String>,
    /// When true, pass the full document text as context (budget-aware).
    pub needs_full_document: bool,
    /// When true, use outline-based context instead of full text.
    pub use_outline_context: bool,
    /// LoRA task ID for adapter selection (empty = base model).
    pub lora_task: String,
    /// Grammar constraint type name ("free_text", "json_schema", "regex").
    pub grammar_type: SkillGrammarType,
    /// Minimum device tier.
    pub min_tier: SkillTier,
    /// Whether this skill can be performed without an LLM (deterministic).
    pub deterministic: bool,
}

/// Grammar constraint type for a skill.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SkillGrammarType {
    /// No constraint — free text output.
    FreeText,
    /// JSON Schema constraint (schema is built dynamically).
    JsonSchema,
    /// Regex constraint (pattern is built dynamically).
    Regex,
}

impl Default for SkillGrammarType {
    fn default() -> Self {
        SkillGrammarType::FreeText
    }
}

// ---------------------------------------------------------------------------
// Prompt builder
// ---------------------------------------------------------------------------

/// Input for building a skill's system and user prompts.
#[derive(Debug, Clone, Default)]
pub struct SkillPromptInput<'a> {
    /// The primary text (selection text, topic, or instruction).
    pub input: &'a str,
    /// Context text (local context, document text, or outline).
    pub context: &'a str,
    /// Optional keywords.
    pub keywords: &'a str,
    /// Sub-variant context string (if a sub-variant was selected).
    pub variant_context: &'a str,
    /// Device tier for tier-aware prompt construction (thinking suppression, conciseness).
    /// When `None`, no tier-specific directives are added (backward compat).
    pub tier: Option<SkillTier>,
}

/// Output of building a skill's prompts.
#[derive(Debug, Clone)]
pub struct SkillPromptOutput {
    pub system: String,
    pub user: String,
}

impl SkillPromptOutput {
    /// Format as ChatML prompt string for llama-server / MLX server.
    ///
    /// Produces:
    /// ```text
    /// <|im_start|>system
    /// {system}
    /// <|im_end|>
    /// <|im_start|>user
    /// {user}
    /// <|im_end|>
    /// <|im_start|>assistant
    /// {response_prefix}
    /// ```
    pub fn to_chatml(&self, response_prefix: Option<&str>) -> String {
        let mut prompt = format!(
            "<|im_start|>system\n{}\n<|im_end|>\n<|im_start|>user\n{}\n<|im_end|>\n<|im_start|>assistant\n",
            self.system, self.user
        );
        if let Some(prefix) = response_prefix {
            if !prefix.is_empty() {
                prompt.push_str(prefix);
            }
        }
        prompt
    }
}

impl SkillDef {
    /// Build the system and user prompts for this skill.
    pub fn build_prompt(&self, input: SkillPromptInput) -> SkillPromptOutput {
        let variant = if input.variant_context.is_empty() {
            ""
        } else {
            input.variant_context
        };

        let keywords_line = if input.keywords.is_empty() {
            String::new()
        } else {
            format!("\nKeywords to cover: \"{}\"", input.keywords)
        };

        let mut output = match self.id.as_str() {
            // --- Read surface -------------------------------------------------
            "doc_summarize" => SkillPromptOutput {
                system: "Summarize the document in 3-5 bullet points (- ). Capture the key points. Output only the bullets.".into(),
                user: format!("Summarize this document:\n\n{}", input.context),
            },
            "doc_key_points" => SkillPromptOutput {
                system: "Extract 3-5 key takeaways as a numbered list (1. 2. 3.). Be concise. Output only the list.".into(),
                user: format!("Document outline and section summaries:\n{}", input.context),
            },
            "doc_extract_actions" => SkillPromptOutput {
                system: "Extract action items as a bullet list starting with \"- \". Be concise. Output only the list.".into(),
                user: format!("Extract action items from:\n\n{}", input.context),
            },
            "doc_extract_dates" => SkillPromptOutput {
                system: "Extract all dates and deadlines as JSON. Format: {\"dates\": [{\"date\": \"string\", \"context\": \"string\"}]}".into(),
                user: format!("Extract dates from:\n\n{}", input.context),
            },
            "doc_readability_score" => SkillPromptOutput {
                system: "Analyze the reading level. Output JSON: {\"grade_level\": number, \"score\": number, \"suggestions\": \"string\"}".into(),
                user: format!("Analyze readability of:\n\n{}", input.context),
            },
            "doc_word_count" => SkillPromptOutput {
                system: "Count words and statistics. Output JSON: {\"words\": number, \"chars\": number, \"paragraphs\": number, \"reading_time_min\": number}".into(),
                user: format!("Analyze:\n\n{}", input.context),
            },
            "doc_find_contradictions" => SkillPromptOutput {
                system: "Find contradictions in the text. Output JSON: {\"contradictions\": [{\"quote1\": \"string\", \"quote2\": \"string\", \"explanation\": \"string\"}]}. If none found, return empty array.".into(),
                user: format!("Check for contradictions:\n\n{}", input.context),
            },

            // --- Edit surface -------------------------------------------------
            "edit_fix_grammar" => SkillPromptOutput {
                system: "Fix spelling and grammar errors only. Keep meaning and style. Output only the corrected text.".into(),
                user: input.input.into(),
            },
            "edit_improve_writing" => SkillPromptOutput {
                system: format!("Improve the text {} . Keep the meaning. Output only the improved text.",
                    if variant.is_empty() { "for clarity and flow" } else { variant }),
                user: input.input.into(),
            },
            "edit_change_tone" => SkillPromptOutput {
                system: format!("Rewrite {} . Keep the meaning. Output only the rewritten text.",
                    if variant.is_empty() { "in a professional tone" } else { variant }),
                user: input.input.into(),
            },
            "edit_simplify" => SkillPromptOutput {
                system: "Simplify the text. Use shorter sentences and simpler words. Output only the simplified text.".into(),
                user: input.input.into(),
            },
            "edit_make_longer" => SkillPromptOutput {
                system: "Expand the text with more detail, examples, and depth. Keep the same style and meaning. Output only the expanded text.".into(),
                user: input.input.into(),
            },
            "edit_make_shorter" => SkillPromptOutput {
                system: "Condense the text to be shorter. Keep all key information. Do not use bullet points. Output only the shortened text.".into(),
                user: input.input.into(),
            },
            "edit_translate_selection" => SkillPromptOutput {
                system: format!("Translate to {}. Output only the translated text.",
                    if variant.is_empty() { "Spanish" } else { variant }),
                user: input.input.into(),
            },
            "edit_translate_document" => SkillPromptOutput {
                system: format!("Translate to {}. Preserve markdown structure (headings, lists, formatting). Output only the translated text.",
                    if variant.is_empty() { "Spanish" } else { variant }),
                user: format!("Translate this section:\n\n{}", input.context),
            },
            "edit_format_document" => SkillPromptOutput {
                system: "Reformat the text into a well-structured markdown document.\nRules: # for title, ## for sections, ### for subsections. Blank line before/after headings. Use - for bullets, 1. 2. for numbered lists. One paragraph per line.\nDo NOT change wording or fix grammar. Preserve all information. Output only the formatted document.".into(),
                user: format!("Format this document:\n\n{}", input.context),
            },
            "edit_improve_document" => SkillPromptOutput {
                system: "Improve the document for clarity, flow, and professionalism. Fix grammar and structure. Keep the meaning.\nMarkdown rules: blank line before/after headings, one heading per line, one paragraph per line.\nUse # for title, ## for sections, - for bullets.\nOutput only the improved document.".into(),
                user: format!("Improve this document:\n\n{}", input.context),
            },
            "edit_custom_instruction" => SkillPromptOutput {
                system: "You are an editor. Follow the user's instruction to edit the text. Do not explain. Output only the result.".into(),
                user: format!("Instruction: {}\nText: \"{}\"", input.context, input.input),
            },
            "edit_continue_writing" => SkillPromptOutput {
                system: "Continue the text naturally. Keep the same style. Write 2-3 sentences. Output only the continuation.".into(),
                user: input.context.into(),
            },
            "edit_rewrite_section" => SkillPromptOutput {
                system: "Rewrite the section for clarity and flow. Keep the meaning. If an instruction is provided, follow it. Output only the rewritten section.".into(),
                user: {
                    let mut u = format!("Section:\n\n{}", input.input);
                    if !input.context.is_empty() {
                        u.push_str(&format!("\n\nDocument outline:\n{}", input.context));
                    }
                    u
                },
            },

            // --- Create surface -----------------------------------------------
            "create_brainstorm" => SkillPromptOutput {
                system: "Generate 5-7 creative ideas as bullet points (- ). Be specific and actionable. Output only the bullets.".into(),
                user: format!("Brainstorm ideas for: {}", input.input),
            },
            "create_outline" => SkillPromptOutput {
                system: "Generate a document outline as markdown headings (# for title, ## for sections, ### for subsections). Use bullet points (-) for details under each section. Cover the given keywords. Output only the outline.".into(),
                user: format!("Create an outline for: {}{}", input.input, keywords_line),
            },
            "create_write_section" => SkillPromptOutput {
                system: "Write a document section following the instruction. Cover the keywords naturally. If an outline is provided, fit coherently and do not duplicate other sections. Write 2-4 paragraphs. Output only the content, no headings.".into(),
                user: {
                    let mut u = format!("Instruction: {}{}", input.input, keywords_line);
                    if !input.context.is_empty() {
                        u.push_str(&format!("\nFull document outline:\n{}", input.context));
                    }
                    u
                },
            },
            "create_generate_document" => SkillPromptOutput {
                system: "Generate a complete document based on the brief. Use markdown headings (##) for sections and (-) for bullet lists. Cover the keywords. Write clear, well-structured content. Output only the document.".into(),
                user: format!("Write a document about: {}{}", input.input, keywords_line),
            },
            "create_write_intro" => SkillPromptOutput {
                system: "Write an engaging introduction paragraph for the document. Hook the reader and preview the key topics based on the outline. Output only the introduction, no heading.".into(),
                user: format!("Document outline and section summaries:\n{}", input.context),
            },
            "create_write_conclusion" => SkillPromptOutput {
                system: "Write a conclusion paragraph that summarizes the key points and provides closure. Output only the conclusion, no heading.".into(),
                user: format!("Document outline and section summaries:\n{}", input.context),
            },
            "create_suggest_title" => SkillPromptOutput {
                system: "Generate a single concise, engaging title for the document. No quotes. Output only the title.".into(),
                user: format!("Document outline and section summaries:\n{}", input.context),
            },
            "create_email_draft" => SkillPromptOutput {
                system: format!("Write a {} email to {} about {}. Include a subject line and body. Output only the email.",
                    if variant.is_empty() { "professional" } else { variant },
                    if input.context.is_empty() { "the recipient" } else { input.context },
                    input.input),
                user: "Write the email.".into(),
            },
            "create_meeting_agenda" => SkillPromptOutput {
                system: "Generate a structured meeting agenda with time slots and discussion topics. Use markdown headings and bullet points. Output only the agenda.".into(),
                user: format!("Meeting topic: {}{}", input.input,
                    if input.context.is_empty() { String::new() }
                    else { format!("\nAttendees: {}", input.context) }),
            },
            "create_job_description" => SkillPromptOutput {
                system: "Generate a professional job description with responsibilities, requirements, and benefits sections. Use markdown headings. Output only the job description.".into(),
                user: format!("Job title: {}{}\nKey requirements: {}",
                    input.input, keywords_line, input.context),
            },
            "create_press_release" => SkillPromptOutput {
                system: "Write a formal press release. Include a headline, dateline, introduction, body paragraphs, and boilerplate. Use standard press release format. Output only the press release.".into(),
                user: format!("Announcement: {}\nCompany: {}", input.input, input.context),
            },
            "create_social_post" => SkillPromptOutput {
                system: format!("Write a social media post for {}. Tone: {}. Keep it engaging and platform-appropriate. Output only the post.",
                    if variant.is_empty() { "general audience" } else { variant },
                    if input.context.is_empty() { "casual" } else { input.context }),
                user: format!("Topic: {}{}", input.input, keywords_line),
            },
            "create_seo_meta" => SkillPromptOutput {
                system: "Generate SEO metadata. Output JSON: {\"title\": \"string (max 60 chars)\", \"description\": \"string (max 160 chars)\"}".into(),
                user: format!("Page topic: {}\nTarget keywords: {}", input.input,
                    if input.context.is_empty() { "none" } else { input.context }),
            },

            _ => SkillPromptOutput {
                system: "Process the following text.".into(),
                user: input.input.into(),
            },
        };

        // Apply tier-aware thinking suppression directives.
        // Qwen3/Bonsai models support `/no_think` and `/think` chat directives
        // that control whether the model emits `<think>...</think>` blocks.
        // Thinking tokens consume output budget and cause empty outputs on
        // small max_tokens skills, so we suppress thinking for most skills.
        if let Some(tier) = input.tier {
            let should_suppress = match tier {
                SkillTier::Low | SkillTier::Medium => true,
                SkillTier::High => !SkillTier::allows_thinking_for(&self.id),
            };
            if should_suppress {
                output.system = format!("/no_think\n{}", output.system);
            }
        }

        output
    }

    /// Estimate the token cost of this skill's system prompt.
    pub fn system_prompt_token_estimate(&self) -> usize {
        let output = self.build_prompt(SkillPromptInput::default());
        estimate_tokens(&output.system)
    }

    /// Get the stop sequences (defaults to ChatML end token if none specified).
    pub fn effective_stop(&self) -> &[String] {
        if self.stop.is_empty() {
            &EMPTY_STOP
        } else {
            &self.stop
        }
    }

    /// Clamp `max_tokens` to the tier's output cap.
    /// For High tier skills that allow thinking, boost by 2x to accommodate
    /// thinking tokens (the actual answer comes after `</think>`).
    pub fn effective_max_tokens(&self, tier: SkillTier) -> usize {
        let (_, tier_max) = tier.output_cap();
        let base = self.max_tokens.min(tier_max);
        if tier == SkillTier::High && SkillTier::allows_thinking_for(&self.id) {
            (base * 2).min(tier_max * 2)
        } else {
            base
        }
    }

    /// Build the full ChatML-formatted prompt string for this skill.
    /// Combines `build_prompt` (with tier-aware thinking suppression) and
    /// `to_chatml` (with response prefix) into a single ready-to-send string.
    pub fn build_chatml_prompt(&self, input: SkillPromptInput) -> String {
        let output = self.build_prompt(input);
        output.to_chatml(self.response_prefix.as_deref())
    }
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/// Registry of all document AI skills.
pub struct SkillRegistry {
    skills: Vec<SkillDef>,
    by_id: std::collections::HashMap<String, usize>,
}

impl SkillRegistry {
    /// Create a new registry with all 33 document skills pre-registered.
    pub fn new() -> Self {
        let skills = all_skills();
        let by_id = skills
            .iter()
            .enumerate()
            .map(|(i, s)| (s.id.clone(), i))
            .collect();
        Self { skills, by_id }
    }

    /// Get a skill by ID.
    pub fn get(&self, id: &str) -> Option<&SkillDef> {
        self.by_id.get(id).map(|&i| &self.skills[i])
    }

    /// Get all skills.
    pub fn all(&self) -> &[SkillDef] {
        &self.skills
    }

    /// Get skills by surface.
    pub fn by_surface(&self, surface: SkillSurface) -> Vec<&SkillDef> {
        self.skills.iter().filter(|s| s.surface == surface).collect()
    }

    /// Get skills by group.
    pub fn by_group(&self, group: SkillGroup) -> Vec<&SkillDef> {
        self.skills.iter().filter(|s| s.group == group).collect()
    }

    /// Get skills by scope.
    pub fn by_scope(&self, scope: SkillScope) -> Vec<&SkillDef> {
        self.skills.iter().filter(|s| s.scope == scope).collect()
    }

    /// Get all one-click skills.
    pub fn one_click_skills(&self) -> Vec<&SkillDef> {
        self.skills.iter().filter(|s| s.mode == SkillMode::OneClick).collect()
    }

    /// Get all skills that need additional input.
    pub fn input_skills(&self) -> Vec<&SkillDef> {
        self.skills
            .iter()
            .filter(|s| s.mode != SkillMode::OneClick)
            .collect()
    }

    /// Get all multi-step skills.
    pub fn multi_step_skills(&self) -> Vec<&SkillDef> {
        self.skills.iter().filter(|s| s.mode == SkillMode::MultiStep).collect()
    }

    /// Get all deterministic skills (no LLM needed).
    pub fn deterministic_skills(&self) -> Vec<&SkillDef> {
        self.skills.iter().filter(|s| s.deterministic).collect()
    }

    /// Number of registered skills.
    pub fn len(&self) -> usize {
        self.skills.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }
}

impl Default for SkillRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Token estimation
// ---------------------------------------------------------------------------

/// Conservative token estimate (~3 chars/token for English text).
pub fn estimate_tokens(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }
    (text.len() + 2) / 3
}

const EMPTY_STOP: [String; 0] = [];

// ---------------------------------------------------------------------------
// Skill definitions
// ---------------------------------------------------------------------------

fn all_skills() -> Vec<SkillDef> {
    vec![
        // === Read Surface (7 skills) =========================================
        SkillDef {
            id: "doc_summarize".into(),
            label: "Summarize Document".into(),
            description: "Summarize the entire document in bullet points".into(),
            icon: "FileText".into(),
            surface: SkillSurface::Read,
            group: SkillGroup::Document,
            scope: SkillScope::Document,
            mode: SkillMode::OneClick,
            max_tokens: 200,
            temperature: 0.3,
            stop: vec!["<|im_end|>".into()],
            response_prefix: Some("- ".into()),
            sub_variants: vec![],
            needs_topic: false,
            supports_keywords: false,
            topic_label: None,
            needs_full_document: true,
            use_outline_context: false,
            lora_task: "summarize".into(),
            grammar_type: SkillGrammarType::FreeText,
            min_tier: SkillTier::Low,
            deterministic: false,
        },
        SkillDef {
            id: "doc_key_points".into(),
            label: "Key Points".into(),
            description: "Extract 3-5 key takeaways from the document".into(),
            icon: "KeyRound".into(),
            surface: SkillSurface::Read,
            group: SkillGroup::Extract,
            scope: SkillScope::Document,
            mode: SkillMode::OneClick,
            max_tokens: 200,
            temperature: 0.3,
            stop: vec!["<|im_end|>".into()],
            response_prefix: Some("1. ".into()),
            sub_variants: vec![],
            needs_topic: false,
            supports_keywords: false,
            topic_label: None,
            needs_full_document: true,
            use_outline_context: true,
            lora_task: "key_points".into(),
            grammar_type: SkillGrammarType::FreeText,
            min_tier: SkillTier::Low,
            deterministic: false,
        },
        SkillDef {
            id: "doc_extract_actions".into(),
            label: "Extract Action Items".into(),
            description: "Extract action items as a list".into(),
            icon: "ListChecks".into(),
            surface: SkillSurface::Read,
            group: SkillGroup::Extract,
            scope: SkillScope::Document,
            mode: SkillMode::OneClick,
            max_tokens: 150,
            temperature: 0.3,
            stop: vec!["<|im_end|>".into()],
            response_prefix: Some("- ".into()),
            sub_variants: vec![],
            needs_topic: false,
            supports_keywords: false,
            topic_label: None,
            needs_full_document: true,
            use_outline_context: false,
            lora_task: "extract_info".into(),
            grammar_type: SkillGrammarType::FreeText,
            min_tier: SkillTier::Low,
            deterministic: false,
        },
        SkillDef {
            id: "doc_extract_dates".into(),
            label: "Extract Dates & Deadlines".into(),
            description: "Find all dates and deadlines in the document".into(),
            icon: "Calendar".into(),
            surface: SkillSurface::Read,
            group: SkillGroup::Extract,
            scope: SkillScope::Document,
            mode: SkillMode::OneClick,
            max_tokens: 200,
            temperature: 0.1,
            stop: vec!["<|im_end|>".into()],
            response_prefix: None,
            sub_variants: vec![],
            needs_topic: false,
            supports_keywords: false,
            topic_label: None,
            needs_full_document: true,
            use_outline_context: false,
            lora_task: "extract_info".into(),
            grammar_type: SkillGrammarType::JsonSchema,
            min_tier: SkillTier::Low,
            deterministic: true,
        },
        SkillDef {
            id: "doc_readability_score".into(),
            label: "Reading Level".into(),
            description: "Assess reading difficulty and grade level".into(),
            icon: "BookOpen".into(),
            surface: SkillSurface::Read,
            group: SkillGroup::Extract,
            scope: SkillScope::Document,
            mode: SkillMode::OneClick,
            max_tokens: 150,
            temperature: 0.1,
            stop: vec!["<|im_end|>".into()],
            response_prefix: None,
            sub_variants: vec![],
            needs_topic: false,
            supports_keywords: false,
            topic_label: None,
            needs_full_document: true,
            use_outline_context: false,
            lora_task: "".into(),
            grammar_type: SkillGrammarType::JsonSchema,
            min_tier: SkillTier::Low,
            deterministic: true,
        },
        SkillDef {
            id: "doc_word_count".into(),
            label: "Word Count".into(),
            description: "Word count, reading time, and statistics".into(),
            icon: "Hash".into(),
            surface: SkillSurface::Read,
            group: SkillGroup::Extract,
            scope: SkillScope::Document,
            mode: SkillMode::OneClick,
            max_tokens: 100,
            temperature: 0.0,
            stop: vec!["<|im_end|>".into()],
            response_prefix: None,
            sub_variants: vec![],
            needs_topic: false,
            supports_keywords: false,
            topic_label: None,
            needs_full_document: true,
            use_outline_context: false,
            lora_task: "".into(),
            grammar_type: SkillGrammarType::JsonSchema,
            min_tier: SkillTier::Low,
            deterministic: true,
        },
        SkillDef {
            id: "doc_find_contradictions".into(),
            label: "Consistency Check".into(),
            description: "Find potential contradictions in the document".into(),
            icon: "AlertTriangle".into(),
            surface: SkillSurface::Read,
            group: SkillGroup::Extract,
            scope: SkillScope::Document,
            mode: SkillMode::MultiStep,
            max_tokens: 400,
            temperature: 0.2,
            stop: vec!["<|im_end|>".into()],
            response_prefix: None,
            sub_variants: vec![],
            needs_topic: false,
            supports_keywords: false,
            topic_label: None,
            needs_full_document: true,
            use_outline_context: false,
            lora_task: "summarize".into(),
            grammar_type: SkillGrammarType::JsonSchema,
            min_tier: SkillTier::High,
            deterministic: false,
        },

        // === Edit Surface (13 skills) ========================================
        SkillDef {
            id: "edit_fix_grammar".into(),
            label: "Fix Grammar".into(),
            description: "Fix spelling and grammar".into(),
            icon: "CheckCheck".into(),
            surface: SkillSurface::Edit,
            group: SkillGroup::Refine,
            scope: SkillScope::Selection,
            mode: SkillMode::OneClick,
            max_tokens: 300,
            temperature: 0.2,
            stop: vec!["<|im_end|>".into()],
            response_prefix: None,
            sub_variants: vec![],
            needs_topic: false,
            supports_keywords: false,
            topic_label: None,
            needs_full_document: false,
            use_outline_context: false,
            lora_task: "edit_grammar".into(),
            grammar_type: SkillGrammarType::FreeText,
            min_tier: SkillTier::Low,
            deterministic: false,
        },
        SkillDef {
            id: "edit_improve_writing".into(),
            label: "Improve Writing".into(),
            description: "Improve clarity and readability".into(),
            icon: "Wand2".into(),
            surface: SkillSurface::Edit,
            group: SkillGroup::Refine,
            scope: SkillScope::Selection,
            mode: SkillMode::FormInput,
            max_tokens: 300,
            temperature: 0.3,
            stop: vec!["<|im_end|>".into()],
            response_prefix: None,
            sub_variants: vec![
                SkillSubVariant { id: "clarity".into(), label: "Clarity".into(), context: "for clarity and readability".into() },
                SkillSubVariant { id: "concise".into(), label: "Concise".into(), context: "to be more concise".into() },
                SkillSubVariant { id: "engaging".into(), label: "Engaging".into(), context: "to be more engaging".into() },
            ],
            needs_topic: false,
            supports_keywords: false,
            topic_label: None,
            needs_full_document: false,
            use_outline_context: false,
            lora_task: "edit_style".into(),
            grammar_type: SkillGrammarType::FreeText,
            min_tier: SkillTier::Low,
            deterministic: false,
        },
        SkillDef {
            id: "edit_change_tone".into(),
            label: "Change Tone".into(),
            description: "Rewrite in a different tone".into(),
            icon: "Wand2".into(),
            surface: SkillSurface::Edit,
            group: SkillGroup::Refine,
            scope: SkillScope::Selection,
            mode: SkillMode::FormInput,
            max_tokens: 300,
            temperature: 0.3,
            stop: vec!["<|im_end|>".into()],
            response_prefix: None,
            sub_variants: vec![
                SkillSubVariant { id: "professional".into(), label: "Professional".into(), context: "in a professional tone".into() },
                SkillSubVariant { id: "casual".into(), label: "Casual".into(), context: "in a casual, friendly tone".into() },
                SkillSubVariant { id: "confident".into(), label: "Confident".into(), context: "in a confident, assertive tone".into() },
                SkillSubVariant { id: "friendly".into(), label: "Friendly".into(), context: "in a warm, friendly tone".into() },
                SkillSubVariant { id: "persuasive".into(), label: "Persuasive".into(), context: "in a persuasive tone".into() },
                SkillSubVariant { id: "empathetic".into(), label: "Empathetic".into(), context: "in an empathetic tone".into() },
            ],
            needs_topic: false,
            supports_keywords: false,
            topic_label: None,
            needs_full_document: false,
            use_outline_context: false,
            lora_task: "edit_style".into(),
            grammar_type: SkillGrammarType::FreeText,
            min_tier: SkillTier::Low,
            deterministic: false,
        },
        SkillDef {
            id: "edit_simplify".into(),
            label: "Simplify".into(),
            description: "Simplify the language".into(),
            icon: "Wand2".into(),
            surface: SkillSurface::Edit,
            group: SkillGroup::Refine,
            scope: SkillScope::Selection,
            mode: SkillMode::OneClick,
            max_tokens: 300,
            temperature: 0.3,
            stop: vec!["<|im_end|>".into()],
            response_prefix: None,
            sub_variants: vec![],
            needs_topic: false,
            supports_keywords: false,
            topic_label: None,
            needs_full_document: false,
            use_outline_context: false,
            lora_task: "edit_style".into(),
            grammar_type: SkillGrammarType::FreeText,
            min_tier: SkillTier::Low,
            deterministic: false,
        },
        SkillDef {
            id: "edit_make_longer".into(),
            label: "Make Longer".into(),
            description: "Add more detail and depth".into(),
            icon: "Maximize2".into(),
            surface: SkillSurface::Edit,
            group: SkillGroup::Refine,
            scope: SkillScope::Selection,
            mode: SkillMode::OneClick,
            max_tokens: 400,
            temperature: 0.5,
            stop: vec!["<|im_end|>".into()],
            response_prefix: None,
            sub_variants: vec![],
            needs_topic: false,
            supports_keywords: false,
            topic_label: None,
            needs_full_document: false,
            use_outline_context: false,
            lora_task: "edit_style".into(),
            grammar_type: SkillGrammarType::FreeText,
            min_tier: SkillTier::Low,
            deterministic: false,
        },
        SkillDef {
            id: "edit_make_shorter".into(),
            label: "Make Shorter".into(),
            description: "Condense without losing key info".into(),
            icon: "Minimize2".into(),
            surface: SkillSurface::Edit,
            group: SkillGroup::Refine,
            scope: SkillScope::Selection,
            mode: SkillMode::OneClick,
            max_tokens: 200,
            temperature: 0.3,
            stop: vec!["<|im_end|>".into()],
            response_prefix: None,
            sub_variants: vec![],
            needs_topic: false,
            supports_keywords: false,
            topic_label: None,
            needs_full_document: false,
            use_outline_context: false,
            lora_task: "edit_style".into(),
            grammar_type: SkillGrammarType::FreeText,
            min_tier: SkillTier::Low,
            deterministic: false,
        },
        SkillDef {
            id: "edit_translate_selection".into(),
            label: "Translate".into(),
            description: "Translate to target language".into(),
            icon: "Languages".into(),
            surface: SkillSurface::Edit,
            group: SkillGroup::Refine,
            scope: SkillScope::Selection,
            mode: SkillMode::FormInput,
            max_tokens: 300,
            temperature: 0.2,
            stop: vec!["<|im_end|>".into()],
            response_prefix: None,
            sub_variants: vec![
                SkillSubVariant { id: "spanish".into(), label: "Spanish".into(), context: "Spanish".into() },
                SkillSubVariant { id: "french".into(), label: "French".into(), context: "French".into() },
                SkillSubVariant { id: "german".into(), label: "German".into(), context: "German".into() },
                SkillSubVariant { id: "japanese".into(), label: "Japanese".into(), context: "Japanese".into() },
                SkillSubVariant { id: "chinese".into(), label: "Chinese".into(), context: "Chinese".into() },
                SkillSubVariant { id: "vietnamese".into(), label: "Vietnamese".into(), context: "Vietnamese".into() },
            ],
            needs_topic: false,
            supports_keywords: false,
            topic_label: None,
            needs_full_document: false,
            use_outline_context: false,
            lora_task: "translate".into(),
            grammar_type: SkillGrammarType::FreeText,
            min_tier: SkillTier::Low,
            deterministic: false,
        },
        SkillDef {
            id: "edit_translate_document".into(),
            label: "Translate Document".into(),
            description: "Translate the entire document".into(),
            icon: "Languages".into(),
            surface: SkillSurface::Edit,
            group: SkillGroup::Document,
            scope: SkillScope::Document,
            mode: SkillMode::MultiStep,
            max_tokens: 800,
            temperature: 0.2,
            stop: vec!["<|im_end|>".into()],
            response_prefix: None,
            sub_variants: vec![
                SkillSubVariant { id: "spanish".into(), label: "Spanish".into(), context: "Spanish".into() },
                SkillSubVariant { id: "french".into(), label: "French".into(), context: "French".into() },
                SkillSubVariant { id: "german".into(), label: "German".into(), context: "German".into() },
                SkillSubVariant { id: "japanese".into(), label: "Japanese".into(), context: "Japanese".into() },
                SkillSubVariant { id: "chinese".into(), label: "Chinese".into(), context: "Chinese".into() },
                SkillSubVariant { id: "vietnamese".into(), label: "Vietnamese".into(), context: "Vietnamese".into() },
            ],
            needs_topic: false,
            supports_keywords: false,
            topic_label: None,
            needs_full_document: true,
            use_outline_context: false,
            lora_task: "translate".into(),
            grammar_type: SkillGrammarType::FreeText,
            min_tier: SkillTier::Medium,
            deterministic: false,
        },
        SkillDef {
            id: "edit_format_document".into(),
            label: "Format Document".into(),
            description: "Reformat with proper headings, lists, and structure".into(),
            icon: "AlignLeft".into(),
            surface: SkillSurface::Edit,
            group: SkillGroup::Document,
            scope: SkillScope::Document,
            mode: SkillMode::OneClick,
            max_tokens: 1500,
            temperature: 0.2,
            stop: vec!["<|im_end|>".into()],
            response_prefix: None,
            sub_variants: vec![],
            needs_topic: false,
            supports_keywords: false,
            topic_label: None,
            needs_full_document: true,
            use_outline_context: false,
            lora_task: "edit_format".into(),
            grammar_type: SkillGrammarType::FreeText,
            min_tier: SkillTier::Low,
            deterministic: false,
        },
        SkillDef {
            id: "edit_improve_document".into(),
            label: "Improve Document".into(),
            description: "Review and improve the entire document".into(),
            icon: "Wand2".into(),
            surface: SkillSurface::Edit,
            group: SkillGroup::Document,
            scope: SkillScope::Document,
            mode: SkillMode::OneClick,
            max_tokens: 1500,
            temperature: 0.3,
            stop: vec!["<|im_end|>".into()],
            response_prefix: None,
            sub_variants: vec![],
            needs_topic: false,
            supports_keywords: false,
            topic_label: None,
            needs_full_document: true,
            use_outline_context: false,
            lora_task: "edit_style".into(),
            grammar_type: SkillGrammarType::FreeText,
            min_tier: SkillTier::Low,
            deterministic: false,
        },
        SkillDef {
            id: "edit_custom_instruction".into(),
            label: "Ask AI".into(),
            description: "Custom instruction for selected text".into(),
            icon: "MessageSquare".into(),
            surface: SkillSurface::Edit,
            group: SkillGroup::Refine,
            scope: SkillScope::Selection,
            mode: SkillMode::PromptInput,
            max_tokens: 300,
            temperature: 0.4,
            stop: vec!["<|im_end|>".into()],
            response_prefix: None,
            sub_variants: vec![],
            needs_topic: false,
            supports_keywords: false,
            topic_label: None,
            needs_full_document: false,
            use_outline_context: false,
            lora_task: "".into(),
            grammar_type: SkillGrammarType::FreeText,
            min_tier: SkillTier::Low,
            deterministic: false,
        },
        SkillDef {
            id: "edit_continue_writing".into(),
            label: "Continue Writing".into(),
            description: "Continue from where you left off".into(),
            icon: "Sparkles".into(),
            surface: SkillSurface::Edit,
            group: SkillGroup::Generate,
            scope: SkillScope::Cursor,
            mode: SkillMode::OneClick,
            max_tokens: 150,
            temperature: 0.6,
            stop: vec!["<|im_end|>".into()],
            response_prefix: None,
            sub_variants: vec![],
            needs_topic: false,
            supports_keywords: false,
            topic_label: None,
            needs_full_document: false,
            use_outline_context: false,
            lora_task: "generate_doc".into(),
            grammar_type: SkillGrammarType::FreeText,
            min_tier: SkillTier::Low,
            deterministic: false,
        },
        SkillDef {
            id: "edit_rewrite_section".into(),
            label: "Rewrite Section".into(),
            description: "Rewrite the current section".into(),
            icon: "PenLine".into(),
            surface: SkillSurface::Edit,
            group: SkillGroup::Refine,
            scope: SkillScope::Section,
            mode: SkillMode::PromptInput,
            max_tokens: 800,
            temperature: 0.4,
            stop: vec!["<|im_end|>".into()],
            response_prefix: None,
            sub_variants: vec![],
            needs_topic: false,
            supports_keywords: false,
            topic_label: Some("Instruction (optional)".into()),
            needs_full_document: false,
            use_outline_context: true,
            lora_task: "edit_style".into(),
            grammar_type: SkillGrammarType::FreeText,
            min_tier: SkillTier::Low,
            deterministic: false,
        },

        // === Create Surface (13 skills) ======================================
        SkillDef {
            id: "create_brainstorm".into(),
            label: "Brainstorm Ideas".into(),
            description: "Generate ideas on a topic".into(),
            icon: "Lightbulb".into(),
            surface: SkillSurface::Create,
            group: SkillGroup::Generate,
            scope: SkillScope::Topic,
            mode: SkillMode::PromptInput,
            max_tokens: 600,
            temperature: 0.7,
            stop: vec!["<|im_end|>".into()],
            response_prefix: Some("- ".into()),
            sub_variants: vec![],
            needs_topic: true,
            supports_keywords: false,
            topic_label: Some("Topic".into()),
            needs_full_document: false,
            use_outline_context: false,
            lora_task: "generate_doc".into(),
            grammar_type: SkillGrammarType::FreeText,
            min_tier: SkillTier::Low,
            deterministic: false,
        },
        SkillDef {
            id: "create_outline".into(),
            label: "Generate Outline".into(),
            description: "Create a document outline from a topic".into(),
            icon: "ListTree".into(),
            surface: SkillSurface::Create,
            group: SkillGroup::Generate,
            scope: SkillScope::Topic,
            mode: SkillMode::PromptInput,
            max_tokens: 800,
            temperature: 0.4,
            stop: vec!["<|im_end|>".into()],
            response_prefix: Some("# ".into()),
            sub_variants: vec![],
            needs_topic: true,
            supports_keywords: true,
            topic_label: Some("Topic".into()),
            needs_full_document: false,
            use_outline_context: false,
            lora_task: "generate_doc".into(),
            grammar_type: SkillGrammarType::FreeText,
            min_tier: SkillTier::Low,
            deterministic: false,
        },
        SkillDef {
            id: "create_write_section".into(),
            label: "Write Section".into(),
            description: "Write a section from an instruction".into(),
            icon: "PenLine".into(),
            surface: SkillSurface::Create,
            group: SkillGroup::Generate,
            scope: SkillScope::Topic,
            mode: SkillMode::PromptInput,
            max_tokens: 600,
            temperature: 0.5,
            stop: vec!["<|im_end|>".into()],
            response_prefix: None,
            sub_variants: vec![],
            needs_topic: true,
            supports_keywords: true,
            topic_label: Some("Instruction".into()),
            needs_full_document: true,
            use_outline_context: true,
            lora_task: "generate_doc".into(),
            grammar_type: SkillGrammarType::FreeText,
            min_tier: SkillTier::Low,
            deterministic: false,
        },
        SkillDef {
            id: "create_generate_document".into(),
            label: "Generate Document".into(),
            description: "Generate a full document from a brief".into(),
            icon: "Sparkles".into(),
            surface: SkillSurface::Create,
            group: SkillGroup::Generate,
            scope: SkillScope::Topic,
            mode: SkillMode::PromptInput,
            max_tokens: 1500,
            temperature: 0.5,
            stop: vec!["<|im_end|>".into()],
            response_prefix: Some("# ".into()),
            sub_variants: vec![],
            needs_topic: true,
            supports_keywords: true,
            topic_label: Some("Brief".into()),
            needs_full_document: false,
            use_outline_context: false,
            lora_task: "generate_doc".into(),
            grammar_type: SkillGrammarType::FreeText,
            min_tier: SkillTier::Low,
            deterministic: false,
        },
        SkillDef {
            id: "create_write_intro".into(),
            label: "Write Intro".into(),
            description: "Write an engaging introduction".into(),
            icon: "AlignStartHorizontal".into(),
            surface: SkillSurface::Create,
            group: SkillGroup::Document,
            scope: SkillScope::Document,
            mode: SkillMode::OneClick,
            max_tokens: 400,
            temperature: 0.5,
            stop: vec!["<|im_end|>".into()],
            response_prefix: None,
            sub_variants: vec![],
            needs_topic: false,
            supports_keywords: false,
            topic_label: None,
            needs_full_document: true,
            use_outline_context: true,
            lora_task: "generate_doc".into(),
            grammar_type: SkillGrammarType::FreeText,
            min_tier: SkillTier::Low,
            deterministic: false,
        },
        SkillDef {
            id: "create_write_conclusion".into(),
            label: "Write Conclusion".into(),
            description: "Write a conclusion that summarizes key points".into(),
            icon: "AlignEndHorizontal".into(),
            surface: SkillSurface::Create,
            group: SkillGroup::Document,
            scope: SkillScope::Document,
            mode: SkillMode::OneClick,
            max_tokens: 400,
            temperature: 0.5,
            stop: vec!["<|im_end|>".into()],
            response_prefix: None,
            sub_variants: vec![],
            needs_topic: false,
            supports_keywords: false,
            topic_label: None,
            needs_full_document: true,
            use_outline_context: true,
            lora_task: "generate_doc".into(),
            grammar_type: SkillGrammarType::FreeText,
            min_tier: SkillTier::Low,
            deterministic: false,
        },
        SkillDef {
            id: "create_suggest_title".into(),
            label: "Suggest Title".into(),
            description: "Generate a title for the document".into(),
            icon: "Heading".into(),
            surface: SkillSurface::Create,
            group: SkillGroup::Document,
            scope: SkillScope::Document,
            mode: SkillMode::OneClick,
            max_tokens: 20,
            temperature: 0.6,
            stop: vec!["<|im_end|>".into(), "\n".into()],
            response_prefix: None,
            sub_variants: vec![],
            needs_topic: false,
            supports_keywords: false,
            topic_label: None,
            needs_full_document: true,
            use_outline_context: true,
            lora_task: "".into(),
            grammar_type: SkillGrammarType::FreeText,
            min_tier: SkillTier::Low,
            deterministic: false,
        },
        SkillDef {
            id: "create_email_draft".into(),
            label: "Draft Email".into(),
            description: "Write a professional email".into(),
            icon: "Mail".into(),
            surface: SkillSurface::Create,
            group: SkillGroup::Generate,
            scope: SkillScope::Topic,
            mode: SkillMode::FormInput,
            max_tokens: 400,
            temperature: 0.4,
            stop: vec!["<|im_end|>".into()],
            response_prefix: None,
            sub_variants: vec![
                SkillSubVariant { id: "professional".into(), label: "Professional".into(), context: "professional".into() },
                SkillSubVariant { id: "casual".into(), label: "Casual".into(), context: "casual".into() },
                SkillSubVariant { id: "persuasive".into(), label: "Persuasive".into(), context: "persuasive".into() },
            ],
            needs_topic: true,
            supports_keywords: false,
            topic_label: Some("Purpose".into()),
            needs_full_document: false,
            use_outline_context: false,
            lora_task: "create_email".into(),
            grammar_type: SkillGrammarType::FreeText,
            min_tier: SkillTier::Low,
            deterministic: false,
        },
        SkillDef {
            id: "create_meeting_agenda".into(),
            label: "Meeting Agenda".into(),
            description: "Generate a structured meeting agenda".into(),
            icon: "Users".into(),
            surface: SkillSurface::Create,
            group: SkillGroup::Generate,
            scope: SkillScope::Topic,
            mode: SkillMode::PromptInput,
            max_tokens: 400,
            temperature: 0.3,
            stop: vec!["<|im_end|>".into()],
            response_prefix: Some("# ".into()),
            sub_variants: vec![],
            needs_topic: true,
            supports_keywords: false,
            topic_label: Some("Meeting topic".into()),
            needs_full_document: false,
            use_outline_context: false,
            lora_task: "generate_doc".into(),
            grammar_type: SkillGrammarType::FreeText,
            min_tier: SkillTier::Low,
            deterministic: false,
        },
        SkillDef {
            id: "create_job_description".into(),
            label: "Job Description".into(),
            description: "Generate a professional job description".into(),
            icon: "Briefcase".into(),
            surface: SkillSurface::Create,
            group: SkillGroup::Generate,
            scope: SkillScope::Topic,
            mode: SkillMode::FormInput,
            max_tokens: 800,
            temperature: 0.4,
            stop: vec!["<|im_end|>".into()],
            response_prefix: Some("# ".into()),
            sub_variants: vec![],
            needs_topic: true,
            supports_keywords: true,
            topic_label: Some("Job title".into()),
            needs_full_document: false,
            use_outline_context: false,
            lora_task: "generate_doc".into(),
            grammar_type: SkillGrammarType::FreeText,
            min_tier: SkillTier::Low,
            deterministic: false,
        },
        SkillDef {
            id: "create_press_release".into(),
            label: "Press Release".into(),
            description: "Generate a formal press release".into(),
            icon: "Newspaper".into(),
            surface: SkillSurface::Create,
            group: SkillGroup::Generate,
            scope: SkillScope::Topic,
            mode: SkillMode::FormInput,
            max_tokens: 800,
            temperature: 0.3,
            stop: vec!["<|im_end|>".into()],
            response_prefix: None,
            sub_variants: vec![],
            needs_topic: true,
            supports_keywords: false,
            topic_label: Some("Announcement".into()),
            needs_full_document: false,
            use_outline_context: false,
            lora_task: "create_pr".into(),
            grammar_type: SkillGrammarType::FreeText,
            min_tier: SkillTier::Low,
            deterministic: false,
        },
        SkillDef {
            id: "create_social_post".into(),
            label: "Social Post".into(),
            description: "Write a social media post".into(),
            icon: "Share2".into(),
            surface: SkillSurface::Create,
            group: SkillGroup::Generate,
            scope: SkillScope::Topic,
            mode: SkillMode::FormInput,
            max_tokens: 300,
            temperature: 0.6,
            stop: vec!["<|im_end|>".into()],
            response_prefix: None,
            sub_variants: vec![
                SkillSubVariant { id: "twitter".into(), label: "Twitter/X".into(), context: "Twitter/X (short, punchy, max 280 chars)".into() },
                SkillSubVariant { id: "linkedin".into(), label: "LinkedIn".into(), context: "LinkedIn (professional, engaging)".into() },
                SkillSubVariant { id: "instagram".into(), label: "Instagram".into(), context: "Instagram (casual, with emojis)".into() },
            ],
            needs_topic: true,
            supports_keywords: true,
            topic_label: Some("Topic".into()),
            needs_full_document: false,
            use_outline_context: false,
            lora_task: "create_social".into(),
            grammar_type: SkillGrammarType::FreeText,
            min_tier: SkillTier::Low,
            deterministic: false,
        },
        SkillDef {
            id: "create_seo_meta".into(),
            label: "SEO Meta".into(),
            description: "Generate SEO meta title and description".into(),
            icon: "Search".into(),
            surface: SkillSurface::Create,
            group: SkillGroup::Generate,
            scope: SkillScope::Topic,
            mode: SkillMode::PromptInput,
            max_tokens: 100,
            temperature: 0.3,
            stop: vec!["<|im_end|>".into()],
            response_prefix: None,
            sub_variants: vec![],
            needs_topic: true,
            supports_keywords: false,
            topic_label: Some("Page topic".into()),
            needs_full_document: false,
            use_outline_context: false,
            lora_task: "".into(),
            grammar_type: SkillGrammarType::JsonSchema,
            min_tier: SkillTier::Low,
            deterministic: false,
        },
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_has_33_skills() {
        let registry = SkillRegistry::new();
        assert_eq!(registry.len(), 33);
    }

    #[test]
    fn test_no_duplicate_ids() {
        let registry = SkillRegistry::new();
        let ids: Vec<&str> = registry.all().iter().map(|s| s.id.as_str()).collect();
        let unique: std::collections::HashSet<&str> = ids.iter().copied().collect();
        assert_eq!(ids.len(), unique.len(), "duplicate skill IDs found");
    }

    #[test]
    fn test_surfaces_have_correct_counts() {
        let registry = SkillRegistry::new();
        assert_eq!(registry.by_surface(SkillSurface::Read).len(), 7);
        assert_eq!(registry.by_surface(SkillSurface::Edit).len(), 13);
        assert_eq!(registry.by_surface(SkillSurface::Create).len(), 13);
    }

    #[test]
    fn test_get_skill_by_id() {
        let registry = SkillRegistry::new();
        let skill = registry.get("edit_fix_grammar").unwrap();
        assert_eq!(skill.label, "Fix Grammar");
        assert_eq!(skill.surface, SkillSurface::Edit);
        assert_eq!(skill.scope, SkillScope::Selection);
        assert_eq!(skill.mode, SkillMode::OneClick);
    }

    #[test]
    fn test_get_nonexistent_skill() {
        let registry = SkillRegistry::new();
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_deterministic_skills() {
        let registry = SkillRegistry::new();
        let det = registry.deterministic_skills();
        assert!(!det.is_empty());
        for skill in &det {
            assert!(skill.deterministic);
        }
    }

    #[test]
    fn test_multi_step_skills() {
        let registry = SkillRegistry::new();
        let multi = registry.multi_step_skills();
        assert!(!multi.is_empty());
        for skill in &multi {
            assert_eq!(skill.mode, SkillMode::MultiStep);
        }
    }

    #[test]
    fn test_build_prompt_fix_grammar() {
        let registry = SkillRegistry::new();
        let skill = registry.get("edit_fix_grammar").unwrap();
        let output = skill.build_prompt(SkillPromptInput {
            input: "This are a test.",
            ..Default::default()
        });
        assert!(output.system.contains("Fix spelling"));
        assert_eq!(output.user, "This are a test.");
    }

    #[test]
    fn test_build_prompt_change_tone_with_variant() {
        let registry = SkillRegistry::new();
        let skill = registry.get("edit_change_tone").unwrap();
        let output = skill.build_prompt(SkillPromptInput {
            input: "Hello world",
            variant_context: "in a casual, friendly tone",
            ..Default::default()
        });
        assert!(output.system.contains("casual, friendly tone"));
    }

    #[test]
    fn test_build_prompt_create_outline_with_keywords() {
        let registry = SkillRegistry::new();
        let skill = registry.get("create_outline").unwrap();
        let output = skill.build_prompt(SkillPromptInput {
            input: "AI writing assistants",
            keywords: "efficiency, creativity",
            ..Default::default()
        });
        assert!(output.user.contains("AI writing assistants"));
        assert!(output.user.contains("efficiency, creativity"));
    }

    #[test]
    fn test_build_prompt_doc_summarize() {
        let registry = SkillRegistry::new();
        let skill = registry.get("doc_summarize").unwrap();
        let output = skill.build_prompt(SkillPromptInput {
            context: "This is a long document about AI.",
            ..Default::default()
        });
        assert!(output.system.contains("Summarize"));
        assert!(output.user.contains("This is a long document"));
    }

    #[test]
    fn test_build_prompt_create_seo_meta() {
        let registry = SkillRegistry::new();
        let skill = registry.get("create_seo_meta").unwrap();
        let output = skill.build_prompt(SkillPromptInput {
            input: "AI writing tools",
            context: "writing, AI, productivity",
            ..Default::default()
        });
        assert!(output.system.contains("JSON"));
        assert!(output.user.contains("AI writing tools"));
    }

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("abc"), 1);
        assert_eq!(estimate_tokens("abcdef"), 2);
        assert_eq!(estimate_tokens("abcdefghi"), 3);
    }

    #[test]
    fn test_high_tier_only_skills() {
        let registry = SkillRegistry::new();
        let high_tier: Vec<&SkillDef> = registry
            .all()
            .iter()
            .filter(|s| s.min_tier == SkillTier::High)
            .collect();
        assert!(!high_tier.is_empty());
        for skill in &high_tier {
            assert_eq!(skill.min_tier, SkillTier::High);
        }
    }

    #[test]
    fn test_all_skills_have_stop_sequences() {
        let registry = SkillRegistry::new();
        for skill in registry.all() {
            assert!(!skill.stop.is_empty(), "skill {} has no stop sequences", skill.id);
        }
    }

    #[test]
    fn test_lora_task_mapping() {
        let registry = SkillRegistry::new();
        let grammar = registry.get("edit_fix_grammar").unwrap();
        assert_eq!(grammar.lora_task, "edit_grammar");

        let translate = registry.get("edit_translate_selection").unwrap();
        assert_eq!(translate.lora_task, "translate");

        let summarize = registry.get("doc_summarize").unwrap();
        assert_eq!(summarize.lora_task, "summarize");
    }

    #[test]
    fn test_effective_stop() {
        let registry = SkillRegistry::new();
        let skill = registry.get("edit_fix_grammar").unwrap();
        let stop = skill.effective_stop();
        assert!(!stop.is_empty());
        assert!(stop.iter().any(|s| s == "<|im_end|>"));
    }

    #[test]
    fn test_by_scope() {
        let registry = SkillRegistry::new();
        let selection_skills = registry.by_scope(SkillScope::Selection);
        assert!(!selection_skills.is_empty());
        for skill in &selection_skills {
            assert_eq!(skill.scope, SkillScope::Selection);
        }
    }

    #[test]
    fn test_one_click_skills() {
        let registry = SkillRegistry::new();
        let one_click = registry.one_click_skills();
        assert!(!one_click.is_empty());
        for skill in &one_click {
            assert_eq!(skill.mode, SkillMode::OneClick);
        }
    }

    #[test]
    fn test_input_skills() {
        let registry = SkillRegistry::new();
        let input_skills = registry.input_skills();
        assert!(!input_skills.is_empty());
        for skill in &input_skills {
            assert_ne!(skill.mode, SkillMode::OneClick);
        }
    }
}
