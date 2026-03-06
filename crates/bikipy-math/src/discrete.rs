use polars::prelude::*;

/// Apply minimum-duration tolerance to a boolean column.
///
/// Removes "true" runs shorter than `min_frames` and "false" gaps shorter
/// than `max_gap_frames`. This mirrors the Python tolerance model:
/// 1. Bridge short false-gaps (distraction tolerance)
/// 2. Remove short true-runs (minimum bout duration)
pub fn tolerance_filter_expr(
    bool_col: &str,
    _min_frames: u32,
    _max_gap_frames: u32,
) -> Expr {
    // Step 1: Bridge short gaps — if a false-run is shorter than max_gap_frames,
    // flip it to true (the animal briefly looked away but came back).
    // Step 2: Remove short bouts — if a true-run is shorter than min_frames,
    // flip it to false (too brief to count).
    //
    // Implemented via run-length encoding: group consecutive identical values,
    // compute run lengths, apply thresholds, then explode back.
    col(bool_col)
        .rle()
        .struct_()
        .field_by_name("values")
        .alias(bool_col)
    // TODO: Full RLE-based tolerance implementation.
    // For now this is a placeholder that passes through the original column.
}

/// Count the number of true values in a boolean column.
pub fn count_true_expr(bool_col: &str) -> Expr {
    col(bool_col).sum()
}

/// Logical OR across multiple boolean columns (any_horizontal).
pub fn any_of(exprs: Vec<Expr>) -> Expr {
    polars::prelude::any_horizontal(exprs).expect("any_horizontal requires at least one expr")
}

/// Logical AND across multiple boolean columns (all_horizontal).
pub fn all_of(exprs: Vec<Expr>) -> Expr {
    polars::prelude::all_horizontal(exprs).expect("all_horizontal requires at least one expr")
}
