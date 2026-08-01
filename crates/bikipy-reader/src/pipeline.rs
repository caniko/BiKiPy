use polars::prelude::*;

/// A single transformation step in the augmented data pipeline.
///
/// Each step receives a LazyFrame and returns a transformed LazyFrame.
/// Steps are composable — the pipeline chains them via `fold`.
/// Polars optimizes the entire chain as a single lazy plan before execution.
pub trait PipelineStep: Send + Sync {
    /// Human-readable name for logging / progress reporting.
    fn name(&self) -> &str;

    /// Apply this transformation to the lazy frame.
    fn apply(&self, lf: LazyFrame) -> LazyFrame;
}

/// Builder that chains `PipelineStep`s into a full augmentation pipeline.
///
/// All steps are lazy — Polars fuses them into a single optimized plan,
/// parallelizing across columns and partitions automatically.
pub struct AugmentedBuilder {
    steps: Vec<Box<dyn PipelineStep>>,
}

impl AugmentedBuilder {
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Add a step to the pipeline. Steps execute in insertion order.
    pub fn add_step(mut self, step: impl PipelineStep + 'static) -> Self {
        self.steps.push(Box::new(step));
        self
    }

    /// Build the augmented LazyFrame by chaining all steps.
    /// Does NOT collect — returns a lazy plan for further composition.
    pub fn build(&self, raw: LazyFrame) -> LazyFrame {
        self.steps.iter().fold(raw, |lf, step| {
            tracing::debug!(step = step.name(), "applying pipeline step");
            step.apply(lf)
        })
    }

    /// Build and collect to a materialized DataFrame.
    pub fn build_and_collect(&self, raw: LazyFrame) -> bikipy_core::error::Result<DataFrame> {
        Ok(self.build(raw).collect()?)
    }

    /// Build and sink directly to a Parquet file (streaming, low memory).
    pub fn build_and_sink_parquet(
        &self,
        raw: LazyFrame,
        path: &std::path::Path,
    ) -> bikipy_core::error::Result<()> {
        let lf = self.build(raw);
        let mut df = lf.collect()?;
        let mut file = std::fs::File::create(path)?;
        ParquetWriter::new(&mut file).finish(&mut df)?;
        Ok(())
    }
}

impl Default for AugmentedBuilder {
    fn default() -> Self {
        Self::new()
    }
}
