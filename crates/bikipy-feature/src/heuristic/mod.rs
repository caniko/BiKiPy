pub mod combined;
pub mod helper;
pub mod proximity;
pub mod ray;
pub mod solo;

use polars::prelude::*;
use serde::{Deserialize, Serialize};

/// Summary of a heuristic evaluation: how many frames/seconds the behavior occurred.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeuristicSummary {
    pub name: String,
    pub true_frames: u64,
    pub total_frames: u64,
    pub seconds: f64,
}

/// Core heuristic trait. Each heuristic adds one or more boolean columns
/// to a LazyFrame indicating frames where the behavior is detected.
///
/// The `summary` default method is shared by all heuristics — DRY.
pub trait Heuristic: Send + Sync {
    /// Name of the boolean result column added to the DataFrame.
    fn name(&self) -> &str;

    /// Apply this heuristic to the LazyFrame, adding a boolean result column.
    fn evaluate(&self, lf: LazyFrame) -> LazyFrame;

    /// Compute summary statistics from the evaluated boolean column.
    /// Default implementation shared by ALL heuristics — DRY.
    fn summary(&self, df: &DataFrame, fps: f64) -> HeuristicSummary {
        let col = df.column(self.name());
        let (true_frames, total_frames) = match col {
            Ok(series) => {
                let bool_col = series.bool().unwrap();
                let total = bool_col.len() as u64;
                let trues = bool_col.sum().unwrap_or(0) as u64;
                (trues, total)
            }
            Err(_) => (0, 0),
        };

        HeuristicSummary {
            name: self.name().to_string(),
            true_frames,
            total_frames,
            seconds: true_frames as f64 / fps,
        }
    }
}
