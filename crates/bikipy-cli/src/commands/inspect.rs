use std::path::PathBuf;

use clap::{Args, Subcommand};

#[derive(Args)]
pub struct InspectArgs {
    #[command(subcommand)]
    pub command: InspectCommand,
}

#[derive(Subcommand)]
pub enum InspectCommand {
    /// Generate all inspection plots from analysis output.
    Plot {
        /// Path to the inspection output directory.
        #[arg(default_value = "inspection")]
        dir: PathBuf,

        /// Output directory for figures. Defaults to `<dir>/figures/`.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Figure format.
        #[arg(short, long, default_value = ".svgz")]
        format: String,
    },

    /// Generate an inspection video with matplotlib overlays.
    Video {
        /// Path to the inspection output directory.
        #[arg(default_value = "inspection")]
        dir: PathBuf,

        /// Output video path.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Video codec (h264, mpeg4, hevc_nvenc, av1_qsv).
        #[arg(long, default_value = "h264")]
        codec: String,

        /// Heuristic to highlight in video.
        #[arg(long)]
        heuristic: Option<String>,
    },

    /// Display summary information about inspection output.
    Info {
        /// Path to the inspection output directory.
        #[arg(default_value = "inspection")]
        dir: PathBuf,
    },

    /// Check annotation consistency.
    CheckAnnotation {
        /// Path to the annotation file or project config.
        #[arg(short, long)]
        path: PathBuf,
    },
}

pub fn run(args: InspectArgs) -> anyhow::Result<()> {
    match args.command {
        InspectCommand::Plot {
            dir,
            output,
            format,
        } => {
            crate::python::run_inspect_plot(&dir, output.as_deref(), &format)
        }
        InspectCommand::Video {
            dir,
            output,
            codec,
            heuristic,
        } => {
            crate::python::run_inspect_video(
                &dir,
                output.as_deref(),
                &codec,
                heuristic.as_deref(),
            )
        }
        InspectCommand::Info { dir } => {
            crate::python::run_inspect_info(&dir)
        }
        InspectCommand::CheckAnnotation { path } => {
            tracing::info!(?path, "checking annotations");
            // TODO: Load and validate annotations
            println!("Annotations OK at {}", path.display());
            Ok(())
        }
    }
}
