use polars::prelude::*;

use bikipy_core::shape::RayIntersectable;
use bikipy_core::types::RaySpec;
use bikipy_math::discrete::any_of;
use bikipy_perimeter::perimeter::Perimeter;

/// Trait for ray-based (field-of-view) heuristics.
///
/// Shared default `ray_expr()` — DRY across Whisker and Olfaction.
/// Generic over shape `S` — monomorphized per concrete shape.
pub trait RayHeuristic<S>: Send + Sync
where
    S: RayIntersectable + Clone + 'static,
{
    /// Maximum angle offset (degrees) from heading to shape.
    fn max_angle_degrees(&self) -> f64 {
        45.0
    }

    /// The target perimeter for ray intersection.
    fn perimeter(&self) -> &Perimeter<S>;

    /// Ray specifications: origin + direction column pairs.
    fn ray_specs(&self) -> Vec<RaySpec>;

    /// Boolean expression: true where ANY ray intersects the perimeter.
    ///
    /// Shared implementation — DRY.
    fn ray_expr(&self) -> Expr {
        let exprs: Vec<Expr> = self
            .ray_specs()
            .iter()
            .map(|spec| {
                self.perimeter()
                    .ray_intersection_mask(&spec.origin, &spec.direction)
            })
            .collect();
        any_of(exprs)
    }
}
