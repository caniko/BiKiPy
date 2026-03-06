use polars::prelude::*;

use bikipy_core::shape::Expandable;
use bikipy_core::types::CoordinateColumns;
use bikipy_math::discrete::any_of;
use bikipy_perimeter::perimeter::Perimeter;

/// Trait for proximity-based heuristics.
///
/// Shared default `proximity_expr()` — DRY across BodyProximity, Whisker, Olfaction.
/// Generic over shape `S` — monomorphized per concrete shape type.
pub trait ProximityHeuristic<S>: Send + Sync
where
    S: Expandable + Clone + 'static,
    S::Output: Clone + 'static,
{
    /// Maximum distance (meters) for proximity detection. Default: 5cm.
    fn max_distance_meters(&self) -> f64 {
        0.05
    }

    /// The target perimeter to check proximity against.
    fn perimeter(&self) -> &Perimeter<S>;

    /// Which coordinate column pairs to check.
    fn coordinate_columns(&self) -> Vec<CoordinateColumns>;

    /// Boolean expression: true where ANY of the coordinate columns
    /// are within `max_distance_meters` of the perimeter.
    ///
    /// Shared implementation — DRY. Expands the perimeter once, then
    /// checks confinement for each coordinate pair.
    fn proximity_expr(&self) -> Expr {
        let expanded = self.perimeter().expanded(self.max_distance_meters());
        let exprs: Vec<Expr> = self
            .coordinate_columns()
            .iter()
            .map(|coords| expanded.confinement_mask(coords))
            .collect();
        any_of(exprs)
    }
}
