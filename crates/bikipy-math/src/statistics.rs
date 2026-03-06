use polars::prelude::*;

/// Rolling median filter for smoothing coordinate data.
pub fn median_filter_expr(col_name: &str, window_size: usize) -> Expr {
    col(col_name).rolling_median(RollingOptionsFixedWindow {
        window_size,
        min_periods: 1,
        center: true,
        ..Default::default()
    })
}

/// Linear interpolation to fill null values in a column.
pub fn interpolate_expr(col_name: &str) -> Expr {
    col(col_name).interpolate(InterpolationMethod::Linear)
}

/// Forward-fill nulls (last observation carried forward).
pub fn forward_fill_expr(col_name: &str) -> Expr {
    col(col_name).fill_null_with_strategy(FillNullStrategy::Forward(FillNullLimit::None))
}
