use polars::prelude::*;
use serde::{Deserialize, Serialize};

use bikipy_core::shape::{Expandable, RayIntersectable, Shape};
use bikipy_core::types::CoordinateColumns;

/// A labeled perimeter wrapping any geometric shape.
/// Generic over `S` so the compiler monomorphizes all operations
/// per concrete shape — no dynamic dispatch in hot paths.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Perimeter<S: Shape> {
    pub label: String,
    pub shape: S,
}

impl<S: Shape + Clone + 'static> Perimeter<S> {
    pub fn new(label: impl Into<String>, shape: S) -> Self {
        Self {
            label: label.into(),
            shape,
        }
    }

    /// Polars boolean expression: true where coordinates are inside this perimeter.
    pub fn confinement_mask(&self, coords: &CoordinateColumns) -> Expr {
        self.shape.confinement_expr(&coords.x, &coords.y)
    }

    /// Filter a LazyFrame to only rows inside this perimeter.
    pub fn filter_confined(&self, lf: LazyFrame, coords: &CoordinateColumns) -> LazyFrame {
        lf.filter(self.confinement_mask(coords))
    }
}

impl<S: Expandable + Clone + 'static> Perimeter<S>
where
    S::Output: Clone + 'static,
{
    /// Create an expanded perimeter (for proximity detection).
    pub fn expanded(&self, distance: f64) -> Perimeter<S::Output> {
        Perimeter {
            label: format!("{}_expanded_{distance}", self.label),
            shape: self.shape.expand(distance),
        }
    }
}

impl<S: RayIntersectable + Clone + 'static> Perimeter<S> {
    /// Polars boolean expression: true where a ray from the given origin
    /// in the given direction intersects this perimeter.
    pub fn ray_intersection_mask(
        &self,
        origin: &CoordinateColumns,
        direction: &CoordinateColumns,
    ) -> Expr {
        self.shape
            .ray_filter_expr(&origin.x, &origin.y, &direction.x, &direction.y)
    }
}
