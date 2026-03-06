use polars::prelude::*;
use serde::{Deserialize, Serialize};

use bikipy_core::shape::{Expandable, RayIntersectable};
use bikipy_core::types::{CoordinateColumns, RaySpec};
use bikipy_perimeter::perimeter::Perimeter;

use crate::heuristic::proximity::ProximityHeuristic;
use crate::heuristic::ray::RayHeuristic;
use crate::heuristic::Heuristic;

/// Detects whisker/ear interactions with objects.
///
/// Combines proximity (left/right ear within distance) AND
/// field-of-view (ray from center_ear through left/right ear intersects object).
///
/// Reuses `ProximityHeuristic::proximity_expr()` and `RayHeuristic::ray_expr()`
/// defaults — DRY. Monomorphized per shape type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhiskerHeuristic<S: Expandable + RayIntersectable> {
    pub perimeter: Perimeter<S>,
    pub max_distance: f64,
    pub max_angle: f64,
}

impl<S> ProximityHeuristic<S> for WhiskerHeuristic<S>
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
        vec![
            CoordinateColumns::new("left_ear_x", "left_ear_y"),
            CoordinateColumns::new("right_ear_x", "right_ear_y"),
        ]
    }
}

impl<S> RayHeuristic<S> for WhiskerHeuristic<S>
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
            // Ray from center_ear through left_ear
            RaySpec {
                origin: CoordinateColumns::new("center_ear_x", "center_ear_y"),
                direction: CoordinateColumns::new("left_ear_dx", "left_ear_dy"),
            },
            // Ray from center_ear through right_ear
            RaySpec {
                origin: CoordinateColumns::new("center_ear_x", "center_ear_y"),
                direction: CoordinateColumns::new("right_ear_dx", "right_ear_dy"),
            },
        ]
    }
}

impl<S> Heuristic for WhiskerHeuristic<S>
where
    S: Expandable + RayIntersectable + Clone + Send + Sync + 'static,
    S::Output: Clone + 'static,
{
    fn name(&self) -> &str {
        "whisker_interaction"
    }

    fn evaluate(&self, lf: LazyFrame) -> LazyFrame {
        // Proximity AND ray: both conditions must be true
        let expr = self.proximity_expr().and(self.ray_expr());
        lf.with_column(expr.alias(self.name()))
    }
}
