use polars::prelude::*;
use serde::{Deserialize, Serialize};

use bikipy_core::shape::{Expandable, RayIntersectable};
use bikipy_core::types::{CoordinateColumns, RaySpec};
use bikipy_perimeter::perimeter::Perimeter;

use crate::heuristic::Heuristic;
use crate::heuristic::proximity::ProximityHeuristic;
use crate::heuristic::ray::RayHeuristic;

/// Detects olfactory investigation of objects.
///
/// Combines proximity (nose within distance) AND
/// field-of-view (ray from snout toward object intersects perimeter).
///
/// Reuses shared trait defaults — DRY. Monomorphized per shape type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OlfactionHeuristic<S: Expandable + RayIntersectable> {
    pub perimeter: Perimeter<S>,
    pub max_distance: f64,
    pub max_angle: f64,
}

impl<S> ProximityHeuristic<S> for OlfactionHeuristic<S>
where
    S: Expandable + RayIntersectable + Clone + Send + Sync + 'static,
    S::Output: Clone + 'static,
{
    fn max_distance_meters(&self) -> f64 {
        self.max_distance
    }

    fn perimeter(&self) -> &Perimeter<S> {
        &self.perimeter
    }

    fn coordinate_columns(&self) -> Vec<CoordinateColumns> {
        vec![CoordinateColumns::new("nose_x", "nose_y")]
    }
}

impl<S> RayHeuristic<S> for OlfactionHeuristic<S>
where
    S: Expandable + RayIntersectable + Clone + Send + Sync + 'static,
    S::Output: Clone + 'static,
{
    fn max_angle_degrees(&self) -> f64 {
        self.max_angle
    }

    fn perimeter(&self) -> &Perimeter<S> {
        &self.perimeter
    }

    fn ray_specs(&self) -> Vec<RaySpec> {
        vec![
            // Ray from nose in the direction the snout is pointing
            // (center_ear → nose direction)
            RaySpec {
                origin: CoordinateColumns::new("nose_x", "nose_y"),
                direction: CoordinateColumns::new("snout_dx", "snout_dy"),
            },
        ]
    }
}

impl<S> Heuristic for OlfactionHeuristic<S>
where
    S: Expandable + RayIntersectable + Clone + Send + Sync + 'static,
    S::Output: Clone + 'static,
{
    fn name(&self) -> &str {
        "olfaction"
    }

    fn evaluate(&self, lf: LazyFrame) -> LazyFrame {
        let expr = self.proximity_expr().and(self.ray_expr());
        lf.with_column(expr.alias(self.name()))
    }
}
