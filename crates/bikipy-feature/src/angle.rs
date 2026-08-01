use polars::prelude::*;

use bikipy_math::geometry::signed_angle_expr;
use bikipy_math::vector::direction_exprs;

/// Compute the heading angle of the animal from two body-part coordinate pairs.
/// E.g., center_ear → nose defines the snout direction.
pub fn heading_angle_expr(_from_x: &str, _from_y: &str, _to_x: &str, _to_y: &str) -> Expr {
    signed_angle_expr(
        &format!("_dir_x"),
        &format!("_dir_y"),
        // Reference direction: positive x-axis (1, 0)
        "_ref_x",
        "_ref_y",
    )
}

/// Add direction vector columns for a heading (from → to).
pub fn add_direction_columns(
    lf: LazyFrame,
    from_x: &str,
    from_y: &str,
    to_x: &str,
    to_y: &str,
    prefix: &str,
) -> LazyFrame {
    let exprs = direction_exprs(from_x, from_y, to_x, to_y, prefix);
    let mut lf = lf;
    for expr in exprs {
        lf = lf.with_column(expr);
    }
    lf
}
