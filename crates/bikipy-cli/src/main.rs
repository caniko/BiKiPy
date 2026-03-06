mod commands;
pub(crate) mod python;

use clap::Parser;

/// BikIPY — automatic analysis of kinematic data from behavioral experiments.
#[derive(Parser)]
#[command(name = "bkpy", version, about)]
enum Cli {
    /// Data ingestion and analysis commands.
    Ingress(commands::ingress::IngressArgs),

    /// Inspection and visualization commands.
    Inspect(commands::inspect::InspectArgs),
}

fn main() -> anyhow::Result<()> {
    // Initialize structured logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();
    match cli {
        Cli::Ingress(args) => commands::ingress::run(args),
        Cli::Inspect(args) => commands::inspect::run(args),
    }
}
