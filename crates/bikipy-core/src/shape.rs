use polars::prelude::*;

use crate::types::BoundingBox;

/// Core geometric trait. Monomorphized per concrete shape —
/// `Circle::contains()` compiles to a radius check,
/// `Polygon::contains()` compiles to ray-casting.
pub trait Shape: Send + Sync {
    /// Test whether a single point is inside this shape.
    fn contains(&self, x: f64, y: f64) -> bool;

    /// Axis-aligned bounding box for fast spatial pre-filtering.
    fn bounding_box(&self) -> BoundingBox;

    /// Polars boolean expression: true for rows where (x_col, y_col) is inside this shape.
    ///
    /// Default implementation applies `contains` row-wise via `map_multiple`.
    /// Concrete shapes can override with vectorized implementations.
    fn confinement_expr(&self, x_col: &str, y_col: &str) -> Expr
    where
        Self: Clone + 'static,
    {
        let shape = self.clone();
        let x = x_col.to_string();
        let y = y_col.to_string();

        map_multiple(
            move |columns: &mut [Column]| {
                let xs = columns[0].f64()?;
                let ys = columns[1].f64()?;

                let out: BooleanChunked = xs
                    .into_iter()
                    .zip(ys.into_iter())
                    .map(|(x, y)| match (x, y) {
                        (Some(x), Some(y)) => Some(shape.contains(x, y)),
                        _ => Some(false),
                    })
                    .collect();

                Ok(out.into_column())
            },
            [col(&x), col(&y)],
            |_schema: &Schema, _fields: &[Field]| {
                Ok(Field::new("confinement".into(), DataType::Boolean))
            },
        )
    }
}

/// Shapes that can be offset (expanded or contracted) by a distance.
pub trait Expandable: Shape {
    type Output: Shape;

    /// Create a new shape expanded outward by `distance` meters.
    fn expand(&self, distance: f64) -> Self::Output;
}

/// Shapes that support ray intersection tests for directional filtering.
pub trait RayIntersectable: Shape {
    /// Test whether a ray from `origin` in `direction` intersects this shape.
    fn ray_intersects(&self, origin: (f64, f64), direction: (f64, f64)) -> bool;

    /// Polars boolean expression: true where the ray from origin columns
    /// in direction columns intersects this shape.
    fn ray_filter_expr(&self, origin_x: &str, origin_y: &str, dir_x: &str, dir_y: &str) -> Expr
    where
        Self: Clone + 'static,
    {
        let shape = self.clone();
        let ox = origin_x.to_string();
        let oy = origin_y.to_string();
        let dx = dir_x.to_string();
        let dy = dir_y.to_string();

        map_multiple(
            move |columns: &mut [Column]| {
                let oxs = columns[0].f64()?;
                let oys = columns[1].f64()?;
                let dxs = columns[2].f64()?;
                let dys = columns[3].f64()?;

                let out: BooleanChunked = oxs
                    .into_iter()
                    .zip(oys.into_iter())
                    .zip(dxs.into_iter())
                    .zip(dys.into_iter())
                    .map(|(((ox, oy), dx), dy)| match (ox, oy, dx, dy) {
                        (Some(ox), Some(oy), Some(dx), Some(dy)) => {
                            Some(shape.ray_intersects((ox, oy), (dx, dy)))
                        }
                        _ => Some(false),
                    })
                    .collect();

                Ok(out.into_column())
            },
            [col(&ox), col(&oy), col(&dx), col(&dy)],
            |_schema: &Schema, _fields: &[Field]| {
                Ok(Field::new("ray_filter".into(), DataType::Boolean))
            },
        )
    }
}
