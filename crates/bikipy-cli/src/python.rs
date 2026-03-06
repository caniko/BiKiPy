use std::path::Path;
use std::process::Command;

/// Invoke the `bkpy-inspect` Python CLI as a subprocess.
///
/// This bridges Rust analysis output to Python matplotlib visualization
/// in a single seamless CLI flow. The user never needs to know two
/// languages are involved.
pub fn invoke_bkpy_inspect(args: &[&str]) -> anyhow::Result<()> {
    let status = Command::new("bkpy-inspect")
        .args(args)
        .status();

    match status {
        Ok(s) if s.success() => Ok(()),
        Ok(s) => {
            anyhow::bail!(
                "bkpy-inspect exited with status {}. \
                 Ensure bikipy-inspect is installed: pip install bikipy-inspect",
                s
            );
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            anyhow::bail!(
                "bkpy-inspect not found on PATH. \
                 Install it with: pip install bikipy-inspect"
            );
        }
        Err(e) => Err(e.into()),
    }
}

/// Run `bkpy-inspect plot` on an inspection output directory.
pub fn run_inspect_plot(
    inspection_dir: &Path,
    output_dir: Option<&Path>,
    format: &str,
) -> anyhow::Result<()> {
    let dir_str = inspection_dir.display().to_string();
    let mut args = vec!["plot", &dir_str, "-f", format];

    let out_str;
    if let Some(out) = output_dir {
        out_str = out.display().to_string();
        args.extend(["-o", &out_str]);
    }

    invoke_bkpy_inspect(&args)
}

/// Run `bkpy-inspect video` on an inspection output directory.
pub fn run_inspect_video(
    inspection_dir: &Path,
    output_path: Option<&Path>,
    codec: &str,
    heuristic: Option<&str>,
) -> anyhow::Result<()> {
    let dir_str = inspection_dir.display().to_string();
    let mut args = vec!["video", &dir_str, "--codec", codec];

    let out_str;
    if let Some(out) = output_path {
        out_str = out.display().to_string();
        args.extend(["-o", &out_str]);
    }

    if let Some(h) = heuristic {
        args.extend(["--heuristic", h]);
    }

    invoke_bkpy_inspect(&args)
}

/// Run `bkpy-inspect info` on an inspection output directory.
pub fn run_inspect_info(inspection_dir: &Path) -> anyhow::Result<()> {
    let dir_str = inspection_dir.display().to_string();
    invoke_bkpy_inspect(&["info", &dir_str])
}
