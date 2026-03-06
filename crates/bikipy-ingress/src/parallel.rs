use rayon::prelude::*;

use bikipy_behaviour::AnalysisResult;
use bikipy_core::error::Result;
use bikipy_reader::pipeline::AugmentedBuilder;

use crate::config::IngressConfig;
use crate::workflow::analyze_experiment;

/// Run all experiments in parallel using Rayon.
///
/// Level 1 parallelism: each experiment on its own Rayon thread.
/// Level 2 parallelism: within each experiment, Polars parallelizes
/// the LazyFrame plan across its own thread pool.
pub fn analyze_all_experiments(config: &IngressConfig) -> Result<Vec<AnalysisResult>> {
    // Configure Rayon thread pool based on runtime settings
    if let Some(num_threads) = config.project.runtime.num_threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
            .ok(); // Ignore if already initialized
    }

    let pipeline = AugmentedBuilder::new();
    // TODO: Build pipeline from config (add steps based on experiment settings)

    let results: Vec<Result<Vec<AnalysisResult>>> = config
        .experiments
        .par_iter()
        .map(|experiment| analyze_experiment(experiment, &pipeline))
        .collect();

    // Flatten results, propagating errors
    let mut all_results = Vec::new();
    for result in results {
        all_results.extend(result?);
    }

    Ok(all_results)
}
