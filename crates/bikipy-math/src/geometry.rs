use polars::prelude::*;

use crate::vector::dot_product_expr;

/// Euclidean distance expression between two 2D coordinate column pairs.
pub fn euclidean_distance_expr(x1: &str, y1: &str, x2: &str, y2: &str) -> Expr {
    let dx = col(x1) - col(x2);
    let dy = col(y1) - col(y2);
    (dx.pow(2) + dy.pow(2)).sqrt()
}

/// Convert pixel coordinates to meters.
pub fn pixel_to_meters_expr(col_name: &str, meters_per_pixel: f64) -> Expr {
    col(col_name) * lit(meters_per_pixel)
}

/// Invert the Y axis (for coordinate systems where Y increases downward).
pub fn invert_y_expr(col_name: &str, height: f64) -> Expr {
    lit(height) - col(col_name)
}

/// Compute the perpendicular distance from each point to a line defined
/// by two points (lx1, ly1) → (lx2, ly2). Fully vectorized.
pub fn perpendicular_distance_to_line_expr(
    px: &str,
    py: &str,
    lx1: f64,
    ly1: f64,
    lx2: f64,
    ly2: f64,
) -> Expr {
    let dx = lx2 - lx1;
    let dy = ly2 - ly1;
    let length = (dx * dx + dy * dy).sqrt();

    // |(p - l1) × (l2 - l1)| / |l2 - l1|
    let cross = (col(px) - lit(lx1)) * lit(dy) - (col(py) - lit(ly1)) * lit(dx);
    cross.abs() / lit(length)
}

/// Signed angle expression (radians) between direction vectors.
/// Uses atan2(cross, dot) for signed result.
pub fn signed_angle_expr(x1: &str, y1: &str, x2: &str, y2: &str) -> Expr {
    let cross = col(x1) * col(y2) - col(y1) * col(x2);
    let dot = dot_product_expr(x1, y1, x2, y2);
    cross.arctan2(dot)
}
