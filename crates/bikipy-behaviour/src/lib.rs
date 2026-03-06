pub mod mapping;
pub mod object_recognition;
pub mod radial_arm;
pub mod reward_tracing;

use polars::prelude::*;
use serde::{Deserialize, Serialize};

use bikipy_feature::heuristic::{Heuristic, HeuristicSummary};

/// Result of analyzing a trial with a behavioural task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    pub task_name: String,
    pub summaries: Vec<HeuristicSummary>,
    /// The fully evaluated DataFrame including all heuristic boolean columns
    /// and coordinate data. Used for inspection export.
    #[serde(skip)]
    pub evaluated_df: Option<DataFrame>,
}

/// Trait for behavioural tasks. Each task defines which heuristics to run
/// and how to analyze the data.
///
/// The `analyze` default method is shared by all tasks — DRY.
/// Associated type `EnclosureShape` drives monomorphization of the
/// entire heuristic pipeline per experiment type.
pub trait BehaviouralTask: Send + Sync {
    /// Human-readable task name.
    fn name(&self) -> &str;

    /// Build the set of heuristics for this task.
    fn heuristics(&self) -> Vec<Box<dyn Heuristic>>;

    /// Run all heuristics and collect summaries.
    /// Default implementation — DRY across all tasks.
    fn analyze(&self, lf: LazyFrame, fps: f64) -> AnalysisResult {
        let heuristics = self.heuristics();

        // Apply all heuristics to the lazy frame.
        // Polars fuses them into a single optimized plan.
        let evaluated = heuristics.iter().fold(lf, |lf, h| h.evaluate(lf));

        let df = evaluated.collect().expect("failed to collect heuristic results");

        let summaries = heuristics.iter().map(|h| h.summary(&df, fps)).collect();

        AnalysisResult {
            task_name: self.name().to_string(),
            summaries,
            evaluated_df: Some(df),
        }
    }
}
