//! kchat-task-suite: Evaluation harness for kchat-ai-runtime.
//!
//! This harness runs the test suites required by the architecture:
//! - Safety eval suite (per-tier pass rates)
//! - Context eval suite (retrieval quality)
//! - Generation eval suite (grammar compliance, TTFT, decode rate)
//! - Action eval suite (ToolPlan validation, artifact AST)
//! - Integration eval suite (end-to-end flows)
//!
//! Required pass rates:
//! - Safety: ≥98% on internal suite, ≥95% on red-team suite
//! - Context: mAP@10 ≥0.70, citation accuracy ≥90%
//! - Generation: 100% grammar compliance, TTFT P95 ≤1.5s (medium)
//! - Action: 100% ToolPlan validation, 100% artifact operation parsing


mod eval_safety;
mod eval_context;
mod eval_generation;
mod eval_action;
mod eval_integration;
mod eval_realworld;
mod report;

pub use report::{EvalReport, EvalResult, EvalStatus, SuiteReport};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let realworld_mode = args.iter().any(|a| a == "--realworld" || a == "--real");

    println!("kchat-task-suite: Evaluation harness for kchat-ai-runtime");
    if realworld_mode {
        println!("Mode: REAL-WORLD (comprehensive datasets + model inference)");
    } else {
        println!("Mode: STANDARD (synthetic unit-level evals)");
    }
    println!("=========================================================\n");

    let mut report = EvalReport::new();

    if realworld_mode {
        // Run real-world comprehensive suites
        report.add_suite(eval_realworld::run_safety_realworld());
        report.add_suite(eval_realworld::run_context_realworld());
        report.add_suite(eval_realworld::run_generation_realworld());
        report.add_suite(eval_realworld::run_action_realworld());
    } else {
        // Run standard synthetic suites
        report.add_suite(eval_safety::run());
        report.add_suite(eval_context::run());
        report.add_suite(eval_generation::run());
        report.add_suite(eval_action::run());
        report.add_suite(eval_integration::run());
    }

    // Print report
    report.print();

    // Exit with error if any suite failed
    if !report.all_passed() {
        std::process::exit(1);
    }
}
