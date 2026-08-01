//! Plugin system for data ingestion.
//!
//! Each plugin handles a specific aspect of data loading:
//! - PerimeterPlugin: load perimeter definitions
//! - VideoPlugin: extract video metadata
//! - FramePlugin: handle frame-level operations
//!
//! Plugins are composable and applied during the ingress workflow.

pub trait IngressPlugin: Send + Sync {
    fn name(&self) -> &str;
    fn apply(&self, config: &crate::config::ExperimentConfig) -> bikipy_core::error::Result<()>;
}
