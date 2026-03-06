use polars::prelude::*;

/// Apply temporal tolerance to a boolean heuristic column.
///
/// 1. Bridge short false-gaps (< `max_gap_frames`): if the animal briefly
///    looked away, treat it as continuous.
/// 2. Remove short true-runs (< `min_bout_frames`): too brief to count.
pub fn apply_tolerance(
    lf: LazyFrame,
    bool_col: &str,
    _min_bout_frames: u32,
    _max_gap_frames: u32,
    output_col: &str,
) -> LazyFrame {
    // RLE-based approach: encode runs, apply thresholds, decode
    // This operates on the collected column for accurate run-length analysis
    lf.with_column(
        col(bool_col)
            // TODO: Implement full RLE-based tolerance
            // For now, pass through. The full implementation will:
            // 1. Compute run-length encoding
            // 2. Bridge false-runs shorter than max_gap_frames
            // 3. Remove true-runs shorter than min_bout_frames
            // 4. Decode back to boolean
            .alias(output_col),
    )
}

/// Convert a boolean heuristic column to seconds of interaction.
pub fn bool_to_seconds(df: &DataFrame, bool_col: &str, fps: f64) -> f64 {
    df.column(bool_col)
        .ok()
        .and_then(|s| s.bool().ok())
        .map(|b| b.sum().unwrap_or(0) as f64 / fps)
        .unwrap_or(0.0)
}
