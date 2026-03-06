use polars::prelude::*;

use crate::pipeline::PipelineStep;
use bikipy_math::velocity::null_high_velocity_exprs;

/// Null out coordinate values where frame-to-frame velocity exceeds a threshold.
/// These are typically tracking artifacts (teleportation).
pub struct HighVelocityFilter {
    pub max_velocity: f64,
    pub x_columns: Vec<String>,
    pub y_columns: Vec<String>,
}

impl PipelineStep for HighVelocityFilter {
    fn name(&self) -> &str {
        "high_velocity_filter"
    }

    fn apply(&self, lf: LazyFrame) -> LazyFrame {
        let mut lf = lf;
        for (x_col, y_col) in self.x_columns.iter().zip(self.y_columns.iter()) {
            let exprs = null_high_velocity_exprs(x_col, y_col, self.max_velocity);
            for expr in exprs {
                lf = lf.with_column(expr);
            }
        }
        lf
    }
}
