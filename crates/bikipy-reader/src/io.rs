use std::path::Path;

use polars::prelude::*;

use bikipy_core::error::{BikipyError, Result};

/// Supported input file formats.
#[derive(Debug, Clone, Copy)]
pub enum InputFormat {
    Csv,
    Parquet,
    Hdf5,
}

impl InputFormat {
    /// Detect format from file extension.
    pub fn from_path(path: &Path) -> Result<Self> {
        match path.extension().and_then(|e| e.to_str()) {
            Some("csv") => Ok(Self::Csv),
            Some("parquet") | Some("pq") => Ok(Self::Parquet),
            Some("h5") | Some("hdf5") => Ok(Self::Hdf5),
            other => Err(BikipyError::Data(format!(
                "unsupported file extension: {other:?}"
            ))),
        }
    }
}

pub(crate) fn to_pl_path(path: &Path) -> Result<PlRefPath> {
    PlRefPath::try_from_path(path).map_err(|e| BikipyError::Data(e.to_string()))
}

/// Load a LazyFrame from a supported file format.
pub fn scan_file(path: &Path) -> Result<LazyFrame> {
    let format = InputFormat::from_path(path)?;
    let pl_path = to_pl_path(path)?;
    match format {
        InputFormat::Csv => Ok(LazyCsvReader::new(pl_path).finish()?),
        InputFormat::Parquet => Ok(LazyFrame::scan_parquet(pl_path, Default::default())?),
        InputFormat::Hdf5 => Err(BikipyError::Data(
            "HDF5 support requires the hdf5 feature — use DeepLabCutReader for .h5 files".into(),
        )),
    }
}

/// Write a DataFrame to Parquet with LZ4 compression.
pub fn write_parquet(df: &mut DataFrame, path: &Path) -> Result<()> {
    let mut file = std::fs::File::create(path)?;
    ParquetWriter::new(&mut file)
        .with_compression(ParquetCompression::Lz4Raw)
        .finish(df)?;
    Ok(())
}
