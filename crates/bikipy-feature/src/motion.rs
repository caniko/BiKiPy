use polars::prelude::*;

use bikipy_math::velocity::velocity_expr;

/// Compute total distance traveled from coordinate columns.
pub fn total_distance_expr(x_col: &str, y_col: &str) -> Expr {
    velocity_expr(x_col, y_col).sum()
}

/// Compute instantaneous speed (distance per frame) and add as a new column.
pub fn add_speed_column(lf: LazyFrame, x_col: &str, y_col: &str, output_col: &str) -> LazyFrame {
    lf.with_column(velocity_expr(x_col, y_col).alias(output_col))
}

/// Compute displacement from the starting position.
pub fn displacement_expr(x_col: &str, y_col: &str) -> Expr {
    let first_x = col(x_col).first();
    let first_y = col(y_col).first();
    ((col(x_col) - first_x).pow(2) + (col(y_col) - first_y).pow(2)).sqrt()
}
