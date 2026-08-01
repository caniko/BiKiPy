use polars::prelude::*;

use crate::pipeline::PipelineStep;
use bikipy_core::types::MidpointGroup;

/// Compute midpoints from groups of body-part columns.
/// E.g., "center_ear" as the midpoint of "left_ear" and "right_ear".
pub struct MidpointComputer {
    pub groups: Vec<MidpointGroup>,
}

impl PipelineStep for MidpointComputer {
    fn name(&self) -> &str {
        "midpoint_computer"
    }

    fn apply(&self, lf: LazyFrame) -> LazyFrame {
        let mut lf = lf;
        for group in &self.groups {
            for coord in ["x", "y"] {
                let source_cols: Vec<Expr> = group
                    .source_labels
                    .iter()
                    .map(|label| col(&format!("{label}_{coord}")))
                    .collect();

                let n = source_cols.len() as f64;
                let sum = source_cols
                    .into_iter()
                    .reduce(|a, b| a + b)
                    .expect("midpoint group must have at least one source");

                lf = lf
                    .with_column((sum / lit(n)).alias(&format!("{}_{coord}", group.output_label)));
            }
        }
        lf
    }
}
