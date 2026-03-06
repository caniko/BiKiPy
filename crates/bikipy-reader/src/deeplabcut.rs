use std::path::Path;

use polars::prelude::*;

use bikipy_core::error::Result;
use bikipy_core::types::Label;

/// Reader specialized for DeepLabCut output files.
///
/// DLC files have a multi-level header:
/// - Row 0: scorer name
/// - Row 1: body part labels
/// - Row 2: coordinate type (x, y, likelihood)
///
/// This reader flattens the multi-level columns into `{label}_{coord}` format
/// for use with the augmented pipeline.
pub struct DeepLabCutReader {
    pub labels: Vec<Label>,
}

impl DeepLabCutReader {
    /// Read a DLC CSV file and return a LazyFrame with flattened columns.
    ///
    /// Columns: `{label}_x`, `{label}_y`, `{label}_likelihood` for each body part.
    pub fn read_csv(&self, path: &Path) -> Result<LazyFrame> {
        // Read with header rows skipped, then rename columns
        let df = CsvReadOptions::default()
            .with_skip_rows(2)
            .with_has_header(true)
            .try_into_reader_with_file_path(Some(path.into()))?
            .finish()?;

        // Rename columns from DLC format to flat format
        let mut renamed = df;
        let col_names: Vec<String> = renamed
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();

        // Skip first column (frame index), then group by 3 (x, y, likelihood per label)
        let data_cols = &col_names[1..];
        for (i, label) in self.labels.iter().enumerate() {
            let base = i * 3;
            if base + 2 < data_cols.len() {
                renamed.rename(&data_cols[base], format!("{label}_x").into())?;
                renamed.rename(&data_cols[base + 1], format!("{label}_y").into())?;
                renamed.rename(&data_cols[base + 2], format!("{label}_likelihood").into())?;
            }
        }

        Ok(renamed.lazy())
    }
}
