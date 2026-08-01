use polars::prelude::*;

use bikipy_core::shape::Expandable;
use bikipy_core::types::CoordinateColumns;
use bikipy_perimeter::perimeter::Perimeter;

/// Compute proximity: checks if coordinates are within a distance threshold
/// of a perimeter by expanding the perimeter and testing confinement.
///
/// This is the core axiom used by all proximity-based heuristics.
/// Generic over shape — monomorphized.
pub struct ComputeProximity<S: Expandable> {
    pub perimeter: Perimeter<S>,
    pub max_distance: f64,
}

impl<S> ComputeProximity<S>
where
    S: Expandable + Clone + 'static,
    S::Output: Clone + 'static,
{
    pub fn new(perimeter: Perimeter<S>, max_distance: f64) -> Self {
        Self {
            perimeter,
            max_distance,
        }
    }

    /// Boolean expression: true where coordinates are within distance of the perimeter.
    pub fn is_proximate(&self, coords: &CoordinateColumns) -> Expr {
        let expanded = self.perimeter.expanded(self.max_distance);
        expanded.confinement_mask(coords)
    }

    /// Apply proximity check to a LazyFrame, adding a boolean column.
    pub fn apply(&self, lf: LazyFrame, coords: &CoordinateColumns, output_col: &str) -> LazyFrame {
        lf.with_column(self.is_proximate(coords).alias(output_col))
    }
}
