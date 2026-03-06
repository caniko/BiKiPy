use polars::prelude::*;

use crate::pipeline::PipelineStep;
use bikipy_math::statistics::{interpolate_expr, median_filter_expr};

/// Apply rolling median + interpolation smoothing to coordinate columns.
pub struct MedianSmoother {
    pub window_size: usize,
    pub columns: Vec<String>,
}

impl PipelineStep for MedianSmoother {
    fn name(&self) -> &str {
        "median_smoother"
    }

    fn apply(&self, lf: LazyFrame) -> LazyFrame {
        let mut lf = lf;
        for col_name in &self.columns {
            // First interpolate nulls, then apply median filter
            lf = lf.with_column(interpolate_expr(col_name).alias(col_name));
            lf = lf.with_column(median_filter_expr(col_name, self.window_size).alias(col_name));
        }
        lf
    }
}
