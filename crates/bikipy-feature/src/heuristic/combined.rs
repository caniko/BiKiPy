use polars::prelude::*;

use bikipy_math::discrete::all_of;

use crate::heuristic::{Heuristic, HeuristicSummary};

/// A combined heuristic that chains a primary (solo) heuristic with
/// zero or more helper heuristics. The final result is the AND
/// of all constituent heuristic columns.
pub struct CombinedHeuristic {
    pub name: String,
    pub primary: Box<dyn Heuristic>,
    pub helpers: Vec<Box<dyn Heuristic>>,
}

impl CombinedHeuristic {
    pub fn new(
        name: impl Into<String>,
        primary: impl Heuristic + 'static,
        helpers: Vec<Box<dyn Heuristic>>,
    ) -> Self {
        Self {
            name: name.into(),
            primary: Box::new(primary),
            helpers,
        }
    }
}

impl Heuristic for CombinedHeuristic {
    fn name(&self) -> &str {
        &self.name
    }

    fn evaluate(&self, lf: LazyFrame) -> LazyFrame {
        // Apply primary heuristic
        let mut lf = self.primary.evaluate(lf);

        // Apply each helper
        for helper in &self.helpers {
            lf = helper.evaluate(lf);
        }

        // AND all results together into the combined column
        let mut component_cols = vec![col(self.primary.name())];
        for helper in &self.helpers {
            component_cols.push(col(helper.name()));
        }

        lf.with_column(all_of(component_cols).alias(&self.name))
    }

    fn summary(&self, df: &DataFrame, fps: f64) -> HeuristicSummary {
        // Use the combined column for summary — shared default logic
        let col = df.column(&self.name);
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
            name: self.name.clone(),
            true_frames,
            total_frames,
            seconds: true_frames as f64 / fps,
        }
    }
}
