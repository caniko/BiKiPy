use clap::Parser;

/// Mirrors the CLI enum for testing parse behavior.
#[derive(Parser, Debug)]
#[command(name = "bkpy")]
enum TestCli {
    Ingress(TestIngressArgs),
    Inspect(TestInspectArgs),
}

#[derive(clap::Args, Debug)]
struct TestIngressArgs {
    #[command(subcommand)]
    command: TestIngressCommand,
}

#[derive(clap::Subcommand, Debug)]
enum TestIngressCommand {
    Init {
        #[arg(default_value = ".")]
        path: std::path::PathBuf,
    },
    Analyze {
        #[arg(short, long, default_value = "bikipy_config.toml")]
        config: std::path::PathBuf,
        #[arg(long)]
        inspect_output: Option<std::path::PathBuf>,
    },
    Update {
        #[arg(short, long, default_value = "bikipy_config.toml")]
        config: std::path::PathBuf,
    },
}

#[derive(clap::Args, Debug)]
struct TestInspectArgs {
    #[command(subcommand)]
    command: TestInspectCommand,
}

#[derive(clap::Subcommand, Debug)]
enum TestInspectCommand {
    CheckAnnotation {
        #[arg(short, long)]
        path: std::path::PathBuf,
    },
}

#[test]
fn parse_ingress_init() {
    let cli = TestCli::try_parse_from(["bkpy", "ingress", "init"]).unwrap();
    assert!(matches!(
        cli,
        TestCli::Ingress(TestIngressArgs {
            command: TestIngressCommand::Init { .. }
        })
    ));
}

#[test]
fn parse_ingress_init_with_path() {
    let cli = TestCli::try_parse_from(["bkpy", "ingress", "init", "/my/project"]).unwrap();
    if let TestCli::Ingress(args) = cli {
        if let TestIngressCommand::Init { path } = args.command {
            assert_eq!(path.to_str().unwrap(), "/my/project");
        } else {
            panic!("expected Init");
        }
    }
}

#[test]
fn parse_ingress_analyze() {
    let cli = TestCli::try_parse_from(["bkpy", "ingress", "analyze"]).unwrap();
    assert!(matches!(
        cli,
        TestCli::Ingress(TestIngressArgs {
            command: TestIngressCommand::Analyze { .. }
        })
    ));
}

#[test]
fn parse_ingress_analyze_with_config() {
    let cli =
        TestCli::try_parse_from(["bkpy", "ingress", "analyze", "-c", "my_config.toml"]).unwrap();
    if let TestCli::Ingress(args) = cli {
        if let TestIngressCommand::Analyze { config, .. } = args.command {
            assert_eq!(config.to_str().unwrap(), "my_config.toml");
        } else {
            panic!("expected Analyze");
        }
    }
}

#[test]
fn parse_ingress_analyze_with_inspect_output() {
    let cli = TestCli::try_parse_from([
        "bkpy",
        "ingress",
        "analyze",
        "--inspect-output",
        "/tmp/inspect",
    ])
    .unwrap();
    if let TestCli::Ingress(args) = cli {
        if let TestIngressCommand::Analyze { inspect_output, .. } = args.command {
            assert_eq!(inspect_output.unwrap().to_str().unwrap(), "/tmp/inspect");
        } else {
            panic!("expected Analyze");
        }
    }
}

#[test]
fn parse_inspect_check_annotation() {
    let cli = TestCli::try_parse_from([
        "bkpy",
        "inspect",
        "check-annotation",
        "-p",
        "/data/annotations.json",
    ])
    .unwrap();
    assert!(matches!(
        cli,
        TestCli::Inspect(TestInspectArgs {
            command: TestInspectCommand::CheckAnnotation { .. }
        })
    ));
}

#[test]
fn parse_invalid_command_fails() {
    let result = TestCli::try_parse_from(["bkpy", "nonexistent"]);
    assert!(result.is_err());
}
