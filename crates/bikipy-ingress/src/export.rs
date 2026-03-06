use std::path::Path;

use polars::prelude::*;

use bikipy_behaviour::AnalysisResult;
use bikipy_core::error::Result;
use bikipy_core::inspection::InspectionManifest;

/// Export inspection artifacts for a single analysis result.
///
/// Writes:
/// - `{experiment}_evaluated.parquet` — full DataFrame with coordinates + boolean columns
/// - `{experiment}_inspection.json` — metadata sidecar for the Python inspection package
pub fn export_inspection(
    output_dir: &Path,
    experiment_name: &str,
    result: &AnalysisResult,
    manifest: &InspectionManifest,
) -> Result<()> {
    std::fs::create_dir_all(output_dir)?;

    // Write evaluated DataFrame as Parquet
    if let Some(ref df) = result.evaluated_df {
        let parquet_path = output_dir.join(format!("{experiment_name}_evaluated.parquet"));
        let mut df_clone = df.clone();
        let mut file = std::fs::File::create(&parquet_path)?;
        ParquetWriter::new(&mut file).finish(&mut df_clone)?;
        tracing::info!(?parquet_path, "wrote evaluation parquet");
    }

    // Write inspection manifest JSON
    let manifest_path = output_dir.join(format!("{experiment_name}_inspection.json"));
    manifest.write_json(&manifest_path)?;
    tracing::info!(?manifest_path, "wrote inspection manifest");

    Ok(())
}
