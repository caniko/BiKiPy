use polars::prelude::*;

use bikipy_core::shape::Shape;
use bikipy_core::types::CoordinateColumns;
use bikipy_perimeter::perimeter::Perimeter;

use crate::heuristic::Heuristic;

/// Helper heuristic: true when coordinates are OUTSIDE the object perimeter.
///
/// Used to ensure the animal is investigating from outside, not from within.
/// Generic over shape — monomorphized.
pub struct OutsidePerimeterHeuristic<S: Shape> {
    pub perimeter: Perimeter<S>,
    pub coords: CoordinateColumns,
}

impl<S: Shape + Clone + Send + Sync + 'static> Heuristic for OutsidePerimeterHeuristic<S> {
    fn name(&self) -> &str {
        "outside_perimeter"
    }

    fn evaluate(&self, lf: LazyFrame) -> LazyFrame {
        let inside = self.perimeter.confinement_mask(&self.coords);
        lf.with_column(inside.not().alias(self.name()))
    }
}
