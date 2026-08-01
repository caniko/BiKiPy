use polars::prelude::*;
use serde::{Deserialize, Serialize};

use bikipy_core::shape::Expandable;
use bikipy_core::types::CoordinateColumns;
use bikipy_perimeter::perimeter::Perimeter;

use crate::heuristic::Heuristic;
use crate::heuristic::proximity::ProximityHeuristic;

/// Detects when center_ear or tail_base are within proximity of an object.
///
/// Proximity-only heuristic — no ray check needed.
/// Generic over shape `S` — monomorphized per concrete shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BodyProximityHeuristic<S: Expandable> {
    pub perimeter: Perimeter<S>,
    pub max_distance: f64,
}

impl<S> ProximityHeuristic<S> for BodyProximityHeuristic<S>
where
    S: Expandable + Clone + Send + Sync + 'static,
    S::Output: Clone + 'static,
{
    fn max_distance_meters(&self) -> f64 {
        self.max_distance
    }

    fn perimeter(&self) -> &Perimeter<S> {
        &self.perimeter
    }

    fn coordinate_columns(&self) -> Vec<CoordinateColumns> {
        vec![
            CoordinateColumns::new("center_ear_x", "center_ear_y"),
            CoordinateColumns::new("tail_base_x", "tail_base_y"),
        ]
    }
}

impl<S> Heuristic for BodyProximityHeuristic<S>
where
    S: Expandable + Clone + Send + Sync + 'static,
    S::Output: Clone + 'static,
{
    fn name(&self) -> &str {
        "body_proximity"
    }

    fn evaluate(&self, lf: LazyFrame) -> LazyFrame {
        lf.with_column(self.proximity_expr().alias(self.name()))
    }
}
