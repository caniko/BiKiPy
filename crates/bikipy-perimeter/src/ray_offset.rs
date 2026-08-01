use polars::prelude::*;

use bikipy_core::shape::RayIntersectable;
use bikipy_core::types::CoordinateColumns;

/// Filter that checks whether the animal's heading (ray direction)
/// is within a maximum angle offset from the direction toward a target shape.
///
/// Generic over `S` — monomorphized per shape type.
#[derive(Debug, Clone)]
pub struct RayOffsetFilter<S: RayIntersectable> {
    pub shape: S,
    pub max_angle_rad: f64,
}

impl<S: RayIntersectable + Clone + 'static> RayOffsetFilter<S> {
    pub fn new(shape: S, max_angle_degrees: f64) -> Self {
        Self {
            shape,
            max_angle_rad: max_angle_degrees.to_radians(),
        }
    }

    /// Boolean expression: true where the ray from `origin` in `heading_direction`
    /// intersects the shape AND the angle between heading and the direction
    /// toward the shape is within `max_angle_rad`.
    pub fn filter_expr(
        &self,
        origin: &CoordinateColumns,
        heading_direction: &CoordinateColumns,
    ) -> Expr {
        // Check ray intersection with shape
        self.shape.ray_filter_expr(
            &origin.x,
            &origin.y,
            &heading_direction.x,
            &heading_direction.y,
        )
    }
}
