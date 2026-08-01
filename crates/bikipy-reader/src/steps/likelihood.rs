use polars::prelude::*;

use crate::pipeline::PipelineStep;

/// Filter rows where the likelihood of tracked body parts falls below a threshold.
/// Sets coordinates to null where likelihood is too low.
pub struct LikelihoodFilter {
    pub threshold: f64,
    pub likelihood_columns: Vec<String>,
    pub coordinate_columns: Vec<String>,
}

impl LikelihoodFilter {
    pub fn new(
        threshold: f64,
        likelihood_columns: Vec<String>,
        coordinate_columns: Vec<String>,
    ) -> Self {
        Self {
            threshold,
            likelihood_columns,
            coordinate_columns,
        }
    }
}

impl PipelineStep for LikelihoodFilter {
    fn name(&self) -> &str {
        "likelihood_filter"
    }

    fn apply(&self, lf: LazyFrame) -> LazyFrame {
        // For each likelihood column, null out the corresponding coordinate
        // columns where likelihood < threshold.
        let mut lf = lf;
        for (lik_col, coord_col) in self
            .likelihood_columns
            .iter()
            .zip(self.coordinate_columns.iter())
        {
            lf = lf.with_column(
                when(col(lik_col).lt(lit(self.threshold)))
                    .then(lit(NULL))
                    .otherwise(col(coord_col))
                    .alias(coord_col),
            );
        }
        lf
    }
}
