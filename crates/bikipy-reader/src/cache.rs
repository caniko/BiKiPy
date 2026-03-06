use std::path::{Path, PathBuf};

use bikipy_core::error::Result;
use polars::prelude::*;

/// Manages caching of augmented DataFrames as Parquet files.
pub struct AugmentedCache {
    pub cache_dir: PathBuf,
}

impl AugmentedCache {
    pub fn new(cache_dir: impl Into<PathBuf>) -> Self {
        Self {
            cache_dir: cache_dir.into(),
        }
    }

    /// Generate the cache file path for a given source file.
    pub fn cache_path(&self, source_path: &Path, suffix: &str) -> PathBuf {
        let stem = source_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");
        self.cache_dir
            .join(format!("{stem}_{suffix}_augmented.parquet"))
    }

    /// Try to load a cached augmented DataFrame.
    pub fn load(&self, source_path: &Path, suffix: &str) -> Result<Option<LazyFrame>> {
        let cache_path = self.cache_path(source_path, suffix);
        if cache_path.exists() {
            tracing::info!(?cache_path, "loading from cache");
            let pl_path = crate::io::to_pl_path(&cache_path)?;
            Ok(Some(LazyFrame::scan_parquet(pl_path, Default::default())?))
        } else {
            Ok(None)
        }
    }

    /// Save an augmented DataFrame to cache.
    pub fn save(&self, source_path: &Path, suffix: &str, df: &mut DataFrame) -> Result<()> {
        std::fs::create_dir_all(&self.cache_dir)?;
        let cache_path = self.cache_path(source_path, suffix);
        tracing::info!(?cache_path, "saving to cache");
        crate::io::write_parquet(df, &cache_path)
    }
}
