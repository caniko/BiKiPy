use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use bikipy_core::config::ProjectConfig;
use bikipy_core::error::{BikipyError, Result};

/// Full ingress configuration loaded from a project TOML file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngressConfig {
    pub project: ProjectConfig,
    pub experiments: Vec<ExperimentConfig>,
}

/// Configuration for a single experiment/session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    pub name: String,
    pub data_files: Vec<PathBuf>,
    pub task_type: String,
    pub fps: f64,
    pub meters_per_pixel: f64,
}

impl IngressConfig {
    /// Load from a TOML file.
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        toml::from_str(&content).map_err(|e| BikipyError::Config(e.to_string()))
    }
}
