use std::path::Path;

use bikipy_behaviour::AnalysisResult;
use bikipy_core::error::Result;
use bikipy_core::inspection::*;
use bikipy_reader::io::scan_file;
use bikipy_reader::pipeline::AugmentedBuilder;

use crate::config::ExperimentConfig;

/// Run the full analysis workflow for a single experiment.
///
/// 1. Load raw data
/// 2. Build augmented pipeline
/// 3. Apply heuristics
/// 4. Collect results
pub fn analyze_experiment(
    config: &ExperimentConfig,
    pipeline: &AugmentedBuilder,
) -> Result<Vec<AnalysisResult>> {
    let results = Vec::new();

    for data_file in &config.data_files {
        tracing::info!(?data_file, experiment = %config.name, "processing");

        let raw_lf = scan_file(data_file)?;
        let augmented_lf = pipeline.build(raw_lf);

        // TODO: Wire up BehaviouralTask based on config.task_type
        // let task = mapping::create_task(&config.task_type, ...)?;
        // let result = task.analyze(augmented_lf, config.fps);
        // results.push(result);

        let _ = augmented_lf;
    }

    Ok(results)
}

/// Run analysis for all experiments in a project, exporting results.
/// When `inspect_output` is provided, inspection artifacts (Parquet + JSON)
/// are written to that directory for each experiment.
pub fn run_project(
    config_path: &Path,
    inspect_output: Option<&Path>,
) -> Result<Vec<AnalysisResult>> {
    let config = crate::config::IngressConfig::from_file(config_path)?;
    let all_results = crate::parallel::analyze_all_experiments(&config)?;

    if let Some(output_dir) = inspect_output {
        export_inspection_artifacts(output_dir, &config, &all_results)?;
    }

    Ok(all_results)
}

/// Export inspection artifacts for all analysis results.
fn export_inspection_artifacts(
    output_dir: &Path,
    config: &crate::config::IngressConfig,
    results: &[AnalysisResult],
) -> Result<()> {
    for (i, result) in results.iter().enumerate() {
        let experiment_name = &result.task_name;

        let heuristic_metas: Vec<HeuristicMeta> = result
            .summaries
            .iter()
            .map(|s| HeuristicMeta {
                name: s.name.clone(),
                result_column: s.name.clone(),
                true_frames: s.true_frames,
                total_frames: s.total_frames,
                seconds: s.seconds,
            })
            .collect();

        let experiment_config = config.experiments.get(i);
        let fps = experiment_config.map_or(30.0, |e| e.fps);
        let mpp = experiment_config.map_or(0.001, |e| e.meters_per_pixel);

        let manifest = InspectionManifest {
            video: bikipy_core::video::VideoMetadata {
                fps,
                total_frames: result.summaries.first().map_or(0, |s| s.total_frames),
                resolution: (0, 0), // TODO: populate from video file
                meters_per_pixel: bikipy_core::types::MetersPerPixel(mpp),
            },
            video_path: experiment_config
                .and_then(|e| e.data_files.first())
                .map(|p| p.display().to_string()),
            labels: Vec::new(),
            label_colors: Default::default(),
            coordinate_columns: Default::default(),
            perimeters: Vec::new(),
            heuristics: heuristic_metas,
            settings: InspectionSettings {
                minimum_seconds_tolerance: config.project.runtime.minimum_seconds_tolerance,
                maximum_seconds_distraction: config.project.runtime.maximum_seconds_distraction,
                meters_per_pixel: mpp,
            },
        };

        crate::export::export_inspection(output_dir, experiment_name, result, &manifest)?;
    }

    Ok(())
}
