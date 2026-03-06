use std::path::PathBuf;

use clap::{Args, Subcommand};

#[derive(Args)]
pub struct IngressArgs {
    #[command(subcommand)]
    pub command: IngressCommand,
}

#[derive(Subcommand)]
pub enum IngressCommand {
    /// Initialize a new bikipy project.
    Init {
        /// Directory to initialize the project in.
        #[arg(default_value = ".")]
        path: PathBuf,
    },

    /// Run the full analysis pipeline.
    ///
    /// By default, inspection figures are generated automatically after
    /// analysis completes. Use --no-inspect to skip figure generation.
    Analyze {
        /// Path to the project configuration file.
        #[arg(short, long, default_value = "bikipy_config.toml")]
        config: PathBuf,

        /// Skip inspection figure generation after analysis.
        #[arg(long, default_value_t = false)]
        no_inspect: bool,

        /// Directory for inspection artifacts and figures.
        /// Defaults to `<config_dir>/inspection/`.
        #[arg(long)]
        inspect_dir: Option<PathBuf>,

        /// Figure format for inspection output.
        #[arg(long, default_value = ".svgz")]
        inspect_format: String,
    },

    /// Update project configuration.
    Update {
        /// Path to the project configuration file.
        #[arg(short, long, default_value = "bikipy_config.toml")]
        config: PathBuf,
    },
}

pub fn run(args: IngressArgs) -> anyhow::Result<()> {
    match args.command {
        IngressCommand::Init { path } => {
            tracing::info!(?path, "initializing project");
            // TODO: Scaffold project directory structure
            println!("Initialized bikipy project at {}", path.display());
            Ok(())
        }
        IngressCommand::Analyze {
            config,
            no_inspect,
            inspect_dir,
            inspect_format,
        } => {
            tracing::info!(?config, "running analysis");

            // Resolve inspection output directory
            let default_inspect_dir = config
                .parent()
                .unwrap_or(std::path::Path::new("."))
                .join("inspection");
            let inspect_output = if no_inspect {
                None
            } else {
                Some(inspect_dir.unwrap_or(default_inspect_dir))
            };

            let results = bikipy_ingress::workflow::run_project(
                &config,
                inspect_output.as_deref(),
            )?;
            println!("Analysis complete: {} results", results.len());

            // Seamlessly generate inspection figures via Python
            if let Some(ref dir) = inspect_output {
                println!("Generating inspection figures...");
                crate::python::run_inspect_plot(dir, None, &inspect_format)?;
            }

            Ok(())
        }
        IngressCommand::Update { config } => {
            tracing::info!(?config, "updating project");
            // TODO: Re-scan data directory and update config
            println!("Updated project config at {}", config.display());
            Ok(())
        }
    }
}
