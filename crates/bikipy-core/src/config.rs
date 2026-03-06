use serde::{Deserialize, Serialize};

/// Runtime settings, equivalent to Python's BikipyRuntimeSettings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeSettings {
    /// Minimum duration (seconds) for a behavioral bout to be counted.
    #[serde(default = "default_min_seconds_tolerance")]
    pub minimum_seconds_tolerance: f64,

    /// Maximum gap (seconds) to bridge between two bouts.
    #[serde(default = "default_max_seconds_distraction")]
    pub maximum_seconds_distraction: f64,

    /// Number of Rayon threads for experiment-level parallelism.
    /// `None` means use all available cores.
    #[serde(default)]
    pub num_threads: Option<usize>,
}

fn default_min_seconds_tolerance() -> f64 {
    0.5
}

fn default_max_seconds_distraction() -> f64 {
    1.0 / 3.0
}

impl Default for RuntimeSettings {
    fn default() -> Self {
        Self {
            minimum_seconds_tolerance: default_min_seconds_tolerance(),
            maximum_seconds_distraction: default_max_seconds_distraction(),
            num_threads: None,
        }
    }
}

/// Project-level configuration loaded from TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectConfig {
    pub project_name: String,
    pub data_directory: String,
    #[serde(default)]
    pub runtime: RuntimeSettings,
}
