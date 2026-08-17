//! Shared utility functions and metrics used across all eval modules.
//!
//! Provides:
//! - Text quality helpers (JSON extraction, repetition, sentence count, language detection)
//! - Classification metrics (precision, recall, F1, confusion matrix, per-class breakdown)
//! - Ranking metrics (MRR, NDCG, recall@k, MAP)
//! - Latency percentile calculation (P50, P95, P99)
//! - Statistical helpers (mean, stddev, min, max)

use std::collections::HashMap;

// ===========================================================================
// Text quality helpers
// ===========================================================================

/// Extract a JSON object or array from text that may contain surrounding prose
/// or markdown code fences. Returns the first balanced JSON substring.
pub fn extract_json(text: &str) -> String {
    let mut trimmed = text.trim();
    // Strip markdown code fences
    if trimmed.starts_with("```") {
        if let Some(nl) = trimmed.find('\n') {
            trimmed = trimmed[nl + 1..].trim();
        }
        if let Some(pos) = trimmed.find("```") {
            trimmed = trimmed[..pos].trim();
        }
    }
    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        if serde_json::from_str::<serde_json::Value>(trimmed).is_ok() {
            return trimmed.to_string();
        }
    }
    for (i, c) in trimmed.char_indices() {
        if c == '{' || c == '[' {
            if let Ok(end) = find_json_end(&trimmed[i..]) {
                return trimmed[i..i + end].to_string();
            }
        }
    }
    String::new()
}

/// Find the end index of a balanced JSON expression starting at position 0.
fn find_json_end(s: &str) -> Result<usize, ()> {
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;
    for (i, c) in s.char_indices() {
        if escape { escape = false; continue; }
        if c == '\\' && in_string { escape = true; continue; }
        if c == '"' { in_string = !in_string; continue; }
        if in_string { continue; }
        match c {
            '{' | '[' => depth += 1,
            '}' | ']' => { depth -= 1; if depth == 0 { return Ok(i + 1); } }
            _ => {}
        }
    }
    Err(())
}

/// Check whether text is highly repetitive (e.g. "the the the the the").
pub fn is_repeated(text: &str) -> bool {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 6 { return false; }
    let unique: std::collections::HashSet<&str> = words.iter().copied().collect();
    unique.len() < words.len() / 2
}

/// Count sentences in text.
pub fn count_sentences(text: &str) -> usize {
    text.split(['.', '!', '?', '\n']).filter(|s| s.trim().len() > 3).count()
}

/// Score whether text contains characters consistent with the expected language.
pub fn detect_language_score(text: &str, expected: &str) -> f64 {
    let has_cjk = text.chars().any(|c|
        (c >= '\u{4E00}' && c <= '\u{9FFF}') || (c >= '\u{3040}' && c <= '\u{30FF}')
    );
    let has_vietnamese = text.chars().any(|c| c >= '\u{00C0}' && c <= '\u{024F}');
    match expected {
        "japanese" | "chinese" => if has_cjk { 1.0 } else { 0.0 },
        "vietnamese" => if has_vietnamese || text.contains('đ') || text.contains('ă') { 1.0 } else { 0.5 },
        "spanish" => if text.contains('ñ') || text.contains('¿') || text.contains('á') { 1.0 } else { 0.5 },
        "french" => if text.contains('ç') || text.contains('é') || text.contains('è') { 1.0 } else { 0.5 },
        "german" => if text.contains("ü") || text.contains("ö") || text.contains("ä") || text.contains("ß") { 1.0 } else { 0.5 },
        "korean" => if text.chars().any(|c| c >= '\u{AC00}' && c <= '\u{D7AF}') { 1.0 } else { 0.0 },
        "arabic" => if text.chars().any(|c| c >= '\u{0600}' && c <= '\u{06FF}') { 1.0 } else { 0.0 },
        "hindi" => if text.chars().any(|c| c >= '\u{0900}' && c <= '\u{097F}') { 1.0 } else { 0.0 },
        "thai" => if text.chars().any(|c| c >= '\u{0E00}' && c <= '\u{0E7F}') { 1.0 } else { 0.0 },
        _ => 1.0,
    }
}

/// Check if text contains any content from a list of keywords (case-insensitive).
pub fn contains_any(text: &str, keywords: &[&str]) -> bool {
    let lower = text.to_lowercase();
    keywords.iter().any(|k| lower.contains(&k.to_lowercase()))
}

/// Check if text contains ALL keywords from a list (case-insensitive).
pub fn contains_all(text: &str, keywords: &[&str]) -> bool {
    let lower = text.to_lowercase();
    keywords.iter().all(|k| lower.contains(&k.to_lowercase()))
}

/// Check if text avoids all forbidden terms (case-insensitive).
pub fn contains_none(text: &str, forbidden: &[&str]) -> bool {
    let lower = text.to_lowercase();
    !forbidden.iter().any(|k| lower.contains(&k.to_lowercase()))
}

// ===========================================================================
// Classification metrics
// ===========================================================================

/// A single classification outcome for metrics computation.
#[derive(Debug, Clone)]
pub struct ClassificationOutcome {
    pub predicted: String,
    pub actual: String,
    pub correct: bool,
}

/// Confusion matrix and per-class metrics.
#[derive(Debug, Clone)]
pub struct ClassificationReport {
    pub total: usize,
    pub correct: usize,
    pub accuracy: f64,
    pub per_class: HashMap<String, ClassMetrics>,
    pub macro_precision: f64,
    pub macro_recall: f64,
    pub macro_f1: f64,
    pub weighted_f1: f64,
}

/// Per-class metrics (precision, recall, F1, support).
#[derive(Debug, Clone, Default)]
pub struct ClassMetrics {
    pub tp: usize,
    pub fp: usize,
    pub fn_: usize,
    pub support: usize,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

impl ClassificationReport {
    /// Build a classification report from a list of outcomes.
    pub fn from_outcomes(outcomes: &[ClassificationOutcome]) -> Self {
        let total = outcomes.len();
        let correct = outcomes.iter().filter(|o| o.correct).count();
        let accuracy = if total > 0 { correct as f64 / total as f64 } else { 0.0 };

        // Build per-class tallies.
        let mut per_class: HashMap<String, ClassMetrics> = HashMap::new();
        for o in outcomes {
            let pred = per_class.entry(o.predicted.clone()).or_default();
            if o.correct { pred.tp += 1; } else { pred.fp += 1; }

            let actual = per_class.entry(o.actual.clone()).or_default();
            actual.support += 1;
            if !o.correct { actual.fn_ += 1; }
        }

        // Compute precision/recall/F1 per class.
        for m in per_class.values_mut() {
            m.precision = if m.tp + m.fp > 0 { m.tp as f64 / (m.tp + m.fp) as f64 } else { 0.0 };
            m.recall = if m.tp + m.fn_ > 0 { m.tp as f64 / (m.tp + m.fn_) as f64 } else { 0.0 };
            m.f1 = if m.precision + m.recall > 0.0 {
                2.0 * m.precision * m.recall / (m.precision + m.recall)
            } else { 0.0 };
        }

        // Macro averages (unweighted mean across classes).
        let n = per_class.len().max(1);
        let macro_precision = per_class.values().map(|m| m.precision).sum::<f64>() / n as f64;
        let macro_recall = per_class.values().map(|m| m.recall).sum::<f64>() / n as f64;
        let macro_f1 = per_class.values().map(|m| m.f1).sum::<f64>() / n as f64;

        // Weighted F1 (weighted by support).
        let total_support = per_class.values().map(|m| m.support).sum::<usize>().max(1);
        let weighted_f1 = per_class.values()
            .map(|m| m.f1 * m.support as f64 / total_support as f64)
            .sum::<f64>();

        Self { total, correct, accuracy, per_class, macro_precision, macro_recall, macro_f1, weighted_f1 }
    }

    /// Print the report in a table format.
    pub fn print(&self, title: &str) {
        println!("\n{}", title);
        println!("{}", "-".repeat(title.len().max(60)));
        println!("Overall: {}/{} correct ({:.2}% accuracy)", self.correct, self.total, self.accuracy * 100.0);
        println!("Macro P/R/F1: {:.3} / {:.3} / {:.3}", self.macro_precision, self.macro_recall, self.macro_f1);
        println!("Weighted F1: {:.3}", self.weighted_f1);
        println!();
        println!("  {:<25} {:>6} {:>6} {:>6} {:>6} {:>8} {:>8} {:>8}", "Class", "TP", "FP", "FN", "Supp", "Prec", "Rec", "F1");
        println!("  {}", "-".repeat(85));
        let mut classes: Vec<_> = self.per_class.iter().collect();
        classes.sort_by_key(|(k, _)| *k);
        for (class, m) in &classes {
            println!("  {:<25} {:>6} {:>6} {:>6} {:>6} {:>8.3} {:>8.3} {:>8.3}",
                class, m.tp, m.fp, m.fn_, m.support, m.precision, m.recall, m.f1);
        }
    }
}

// ===========================================================================
// Ranking metrics (for retrieval / search evaluation)
// ===========================================================================

/// Compute Mean Reciprocal Rank (MRR).
/// `ranked_lists` is a list of queries, each with a list of (item_id, is_relevant).
pub fn mrr(ranked_lists: &[Vec<(String, bool)>]) -> f64 {
    if ranked_lists.is_empty() { return 0.0; }
    let sum: f64 = ranked_lists.iter().map(|results| {
        for (i, (_, relevant)) in results.iter().enumerate() {
            if *relevant { return 1.0 / (i + 1) as f64; }
        }
        0.0
    }).sum();
    sum / ranked_lists.len() as f64
}

/// Compute recall@k for a set of ranked lists.
/// `relevant_counts` is the total number of relevant items per query.
pub fn recall_at_k(ranked_lists: &[Vec<(String, bool)>], relevant_counts: &[usize], k: usize) -> f64 {
    if ranked_lists.is_empty() { return 0.0; }
    let sum: f64 = ranked_lists.iter().zip(relevant_counts.iter()).map(|(results, total_relevant)| {
        if *total_relevant == 0 { return 1.0; }
        let found = results.iter().take(k).filter(|(_, rel)| *rel).count();
        found as f64 / *total_relevant as f64
    }).sum();
    sum / ranked_lists.len() as f64
}

/// Compute NDCG@k (Normalized Discounted Cumulative Gain).
/// `relevance_scores` is a list of queries, each with graded relevance per ranked position.
pub fn ndcg_at_k(ranked_lists: &[Vec<f64>], k: usize) -> f64 {
    if ranked_lists.is_empty() { return 0.0; }
    let sum: f64 = ranked_lists.iter().map(|relevances| {
        let dcg: f64 = relevances.iter().take(k).enumerate().map(|(i, &rel)| {
            if rel > 0.0 { (2.0_f64.powf(rel) - 1.0) / (i as f64 + 2.0).log2() } else { 0.0 }
        }).sum();
        let mut ideal = relevances.to_vec();
        ideal.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let idcg: f64 = ideal.iter().take(k).enumerate().map(|(i, &rel)| {
            if rel > 0.0 { (2.0_f64.powf(rel) - 1.0) / (i as f64 + 2.0).log2() } else { 0.0 }
        }).sum();
        if idcg > 0.0 { dcg / idcg } else { 0.0 }
    }).sum();
    sum / ranked_lists.len() as f64
}

/// Compute MAP (Mean Average Precision) for a set of ranked lists.
pub fn map_score(ranked_lists: &[Vec<(String, bool)>]) -> f64 {
    if ranked_lists.is_empty() { return 0.0; }
    let sum: f64 = ranked_lists.iter().map(|results| {
        let relevant_total = results.iter().filter(|(_, rel)| *rel).count();
        if relevant_total == 0 { return 0.0; }
        let mut hits = 0;
        let mut ap_sum = 0.0;
        for (i, (_, rel)) in results.iter().enumerate() {
            if *rel {
                hits += 1;
                ap_sum += hits as f64 / (i + 1) as f64;
            }
        }
        ap_sum / relevant_total as f64
    }).sum();
    sum / ranked_lists.len() as f64
}

// ===========================================================================
// Latency percentiles
// ===========================================================================

/// Compute percentiles from a list of latency values (in microseconds or ms).
/// Returns (p50, p95, p99, min, max, mean).
pub fn latency_percentiles(values: &[u64]) -> (u64, u64, u64, u64, u64, f64) {
    if values.is_empty() { return (0, 0, 0, 0, 0, 0.0); }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let n = sorted.len();
    let p50 = percentile(&sorted, 50);
    let p95 = percentile(&sorted, 95);
    let p99 = percentile(&sorted, 99);
    let min = sorted[0];
    let max = sorted[n - 1];
    let mean = sorted.iter().sum::<u64>() as f64 / n as f64;
    (p50, p95, p99, min, max, mean)
}

/// Compute a single percentile from a sorted slice.
fn percentile(sorted: &[u64], p: u8) -> u64 {
    if sorted.is_empty() { return 0; }
    let idx = ((p as f64 / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Print a latency summary table.
pub fn print_latency_summary(label: &str, values: &[u64]) {
    if values.is_empty() {
        println!("  {}: no samples", label);
        return;
    }
    let (p50, p95, p99, min, max, mean) = latency_percentiles(values);
    println!("  {}: n={} min={}μs p50={}μs p95={}μs p99={}μs max={}μs mean={:.0}μs",
        label, values.len(), min, p50, p95, p99, max, mean);
}

// ===========================================================================
// Statistical helpers
// ===========================================================================

/// Compute mean of a slice of f64.
pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() { return 0.0; }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Compute standard deviation of a slice of f64.
pub fn stddev(values: &[f64]) -> f64 {
    if values.len() < 2 { return 0.0; }
    let m = mean(values);
    let variance = values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    variance.sqrt()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_json_object() {
        let result = extract_json(r#"Some text {"key": "value"} more text"#);
        assert!(result.contains("key"));
    }

    #[test]
    fn test_extract_json_array() {
        let result = extract_json(r#"prefix [1, 2, 3] suffix"#);
        assert_eq!(result, "[1, 2, 3]");
    }

    #[test]
    fn test_extract_json_code_fence() {
        let result = extract_json("```json\n{\"x\": 1}\n```");
        assert!(result.contains("\"x\""));
    }

    #[test]
    fn test_extract_json_empty() {
        assert!(extract_json("no json here").is_empty());
    }

    #[test]
    fn test_is_repeated() {
        assert!(is_repeated("the the the the the the"));
        assert!(!is_repeated("the quick brown fox jumps over"));
    }

    #[test]
    fn test_count_sentences() {
        assert_eq!(count_sentences("Hello world. This is a test! Right?"), 3);
        assert_eq!(count_sentences("One"), 0);
    }

    #[test]
    fn test_detect_language_cjk() {
        assert_eq!(detect_language_score("日本語のテキスト", "japanese"), 1.0);
        assert_eq!(detect_language_score("Hello world", "japanese"), 0.0);
    }

    #[test]
    fn test_contains_helpers() {
        assert!(contains_any("hello world", &["hello", "foo"]));
        assert!(contains_all("hello world foo", &["hello", "world", "foo"]));
        assert!(contains_none("hello world", &["bad", "evil"]));
    }

    #[test]
    fn test_classification_report() {
        let outcomes = vec![
            ClassificationOutcome { predicted: "allow".into(), actual: "allow".into(), correct: true },
            ClassificationOutcome { predicted: "warn".into(), actual: "warn".into(), correct: true },
            ClassificationOutcome { predicted: "allow".into(), actual: "warn".into(), correct: false },
            ClassificationOutcome { predicted: "warn".into(), actual: "warn".into(), correct: true },
        ];
        let report = ClassificationReport::from_outcomes(&outcomes);
        assert_eq!(report.total, 4);
        assert_eq!(report.correct, 3);
        assert!((report.accuracy - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_mrr() {
        let lists = vec![
            vec![("a".into(), false), ("b".into(), true), ("c".into(), false)],
            vec![("d".into(), true), ("e".into(), false)],
        ];
        // Query 1: rank 2 → 1/2 = 0.5
        // Query 2: rank 1 → 1/1 = 1.0
        // MRR = (0.5 + 1.0) / 2 = 0.75
        assert!((mrr(&lists) - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_recall_at_k() {
        let lists = vec![
            vec![("a".into(), true), ("b".into(), false), ("c".into(), true)],
        ];
        let counts = vec![2];
        // k=1: found 1, total 2 → 0.5
        assert!((recall_at_k(&lists, &counts, 1) - 0.5).abs() < 0.01);
        // k=3: found 2, total 2 → 1.0
        assert!((recall_at_k(&lists, &counts, 3) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_ndcg() {
        let lists = vec![vec![3.0, 2.0, 3.0, 0.0, 1.0]];
        let ndcg = ndcg_at_k(&lists, 5);
        assert!(ndcg > 0.0 && ndcg <= 1.0);
    }

    #[test]
    fn test_map_score() {
        let lists = vec![
            vec![("a".into(), true), ("b".into(), false), ("c".into(), true)],
        ];
        // AP = (1/1 + 2/3) / 2 = (1.0 + 0.667) / 2 = 0.833
        assert!((map_score(&lists) - 0.833).abs() < 0.01);
    }

    #[test]
    fn test_latency_percentiles() {
        let values: Vec<u64> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let (p50, p95, p99, min, max, mean) = latency_percentiles(&values);
        // P50 of 1-10 (10 values, index = round(0.5 * 9) = round(4.5) = 5 → value 6)
        assert!(p50 >= 5 && p50 <= 6);  // median is 5 or 6 depending on rounding
        assert_eq!(min, 1);
        assert_eq!(max, 10);
        assert!((mean - 5.5).abs() < 0.1);
        assert!(p95 >= p50);
        assert!(p99 >= p95);
    }

    #[test]
    fn test_mean_stddev() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((mean(&values) - 3.0).abs() < 0.01);
        assert!(stddev(&values) > 1.0 && stddev(&values) < 2.0);
    }
}
