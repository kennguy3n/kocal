//! Evaluation report types.

use std::collections::HashMap;

/// Status of a single eval case.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvalStatus {
    Pass,
    Fail(String),
    Skip(String),
}

/// Result of a single eval case.
#[derive(Debug, Clone)]
pub struct EvalResult {
    pub name: String,
    pub status: EvalStatus,
    pub duration_ms: u64,
    pub metadata: HashMap<String, String>,
}

impl EvalResult {
    pub fn pass(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: EvalStatus::Pass,
            duration_ms: 0,
            metadata: HashMap::new(),
        }
    }

    pub fn fail(name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: EvalStatus::Fail(reason.into()),
            duration_ms: 0,
            metadata: HashMap::new(),
        }
    }

    pub fn skip(name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: EvalStatus::Skip(reason.into()),
            duration_ms: 0,
            metadata: HashMap::new(),
        }
    }

    pub fn is_pass(&self) -> bool {
        matches!(self.status, EvalStatus::Pass)
    }
}

/// Report for a single eval suite.
#[derive(Debug, Clone)]
pub struct SuiteReport {
    pub suite_name: String,
    pub results: Vec<EvalResult>,
    pub required_pass_rate: f64,
}

impl SuiteReport {
    pub fn new(name: impl Into<String>, required_pass_rate: f64) -> Self {
        Self {
            suite_name: name.into(),
            results: Vec::new(),
            required_pass_rate,
        }
    }

    pub fn add(&mut self, result: EvalResult) {
        self.results.push(result);
    }

    pub fn pass_count(&self) -> usize {
        self.results.iter().filter(|r| r.is_pass()).count()
    }

    pub fn total_count(&self) -> usize {
        self.results.len()
    }

    pub fn pass_rate(&self) -> f64 {
        if self.total_count() == 0 {
            return 1.0;
        }
        self.pass_count() as f64 / self.total_count() as f64
    }

    pub fn passed(&self) -> bool {
        self.pass_rate() >= self.required_pass_rate
    }
}

/// Overall evaluation report.
#[derive(Debug, Default)]
pub struct EvalReport {
    pub suites: Vec<SuiteReport>,
}

impl EvalReport {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_suite(&mut self, suite: SuiteReport) {
        self.suites.push(suite);
    }

    pub fn all_passed(&self) -> bool {
        self.suites.iter().all(|s| s.passed())
    }

    pub fn print(&self) {
        println!("Evaluation Report");
        println!("==================\n");

        for suite in &self.suites {
            let status = if suite.passed() { "PASS" } else { "FAIL" };
            println!(
                "[{}] {} — {}/{} passed ({:.1}%, required: {:.1}%)",
                status,
                suite.suite_name,
                suite.pass_count(),
                suite.total_count(),
                suite.pass_rate() * 100.0,
                suite.required_pass_rate * 100.0
            );

            for result in &suite.results {
                let icon = match &result.status {
                    EvalStatus::Pass => "✓",
                    EvalStatus::Fail(reason) => {
                        println!("    ✗ {} — {}", result.name, reason);
                        continue;
                    }
                    EvalStatus::Skip(reason) => {
                        println!("    ⊘ {} — skipped: {}", result.name, reason);
                        continue;
                    }
                };
                println!("    {} {}", icon, result.name);
            }
            println!();
        }

        let overall = if self.all_passed() { "ALL PASSED" } else { "FAILURES DETECTED" };
        println!("Overall: {}", overall);
    }
}
