use polars::prelude::*;

use crate::pipeline::PipelineStep;
use bikipy_core::shape::Shape;
use bikipy_core::types::CoordinateColumns;
use bikipy_perimeter::perimeter::Perimeter;

/// Filter rows to only those where the animal is inside the trial enclosure.
/// Generic over shape — monomorphized per enclosure type.
pub struct ConfinementFilter<S: Shape + Clone + 'static> {
    pub perimeter: Perimeter<S>,
    pub coords: CoordinateColumns,
}

impl<S: Shape + Clone + Send + Sync + 'static> PipelineStep for ConfinementFilter<S> {
    fn name(&self) -> &str {
        "confinement_filter"
    }

    fn apply(&self, lf: LazyFrame) -> LazyFrame {
        self.perimeter.filter_confined(lf, &self.coords)
    }
}
