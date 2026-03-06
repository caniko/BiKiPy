use polars::prelude::*;

/// Compute frame-to-frame velocity from (x, y) coordinate columns.
/// Returns an expression for speed (magnitude of displacement per frame).
pub fn velocity_expr(x_col: &str, y_col: &str) -> Expr {
    let dx = col(x_col) - col(x_col).shift(lit(1));
    let dy = col(y_col) - col(y_col).shift(lit(1));
    (dx.pow(2) + dy.pow(2)).sqrt()
}

/// Boolean mask: true where velocity exceeds `max_velocity`.
/// Used to flag and remove teleportation artifacts.
pub fn high_velocity_mask_expr(x_col: &str, y_col: &str, max_velocity: f64) -> Expr {
    velocity_expr(x_col, y_col).gt(lit(max_velocity))
}

/// Mark high-velocity points as null (for interpolation later).
/// Returns expressions that replace x and y with null where velocity is too high.
pub fn null_high_velocity_exprs(
    x_col: &str,
    y_col: &str,
    max_velocity: f64,
) -> Vec<Expr> {
    let mask = high_velocity_mask_expr(x_col, y_col, max_velocity);
    vec![
        when(mask.clone())
            .then(lit(NULL))
            .otherwise(col(x_col))
            .alias(x_col),
        when(mask)
            .then(lit(NULL))
            .otherwise(col(y_col))
            .alias(y_col),
    ]
}
