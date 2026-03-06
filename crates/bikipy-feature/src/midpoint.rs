use polars::prelude::*;

/// Compute midpoint of two coordinate column pairs and add as new columns.
pub fn midpoint_exprs(
    x1: &str,
    y1: &str,
    x2: &str,
    y2: &str,
    out_x: &str,
    out_y: &str,
) -> Vec<Expr> {
    vec![
        ((col(x1) + col(x2)) / lit(2.0)).alias(out_x),
        ((col(y1) + col(y2)) / lit(2.0)).alias(out_y),
    ]
}
