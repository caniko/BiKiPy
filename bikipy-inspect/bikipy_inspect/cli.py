from __future__ import annotations

from pathlib import Path

import click


@click.group()
def main():
    """bikipy-inspect: Visualization tools for bikipy analysis output."""


@main.command()
@click.argument("inspection_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None, help="Output directory for figures.")
@click.option("--format", "-f", "fmt", type=click.Choice([".svgz", ".jpg", ".png", ".pdf"]), default=".svgz")
def plot(inspection_dir: Path, output: Path | None, fmt: str):
    """Generate all inspection plots from an analysis output directory."""
    from bikipy_inspect.manifest import InspectionManifest
    from bikipy_inspect.plot.generic import save_figure
    from bikipy_inspect.plot.heuristic import (
        plot_heuristic_on_coordinates,
        plot_heuristic_summary,
        plot_heuristic_timeline,
    )

    manifest = InspectionManifest.from_directory(inspection_dir)
    output_dir = output or inspection_dir / "figures"

    # Summary plot
    fig = plot_heuristic_summary(manifest)
    path = save_figure(fig, output_dir / "summary", fmt=fmt)
    click.echo(f"Saved: {path}")

    # Per-heuristic plots
    for h in manifest.heuristics:
        fig = plot_heuristic_timeline(manifest, h.name)
        path = save_figure(fig, output_dir / f"timeline_{h.name}", fmt=fmt)
        click.echo(f"Saved: {path}")

        # Coordinate overlay per label
        for label in manifest.labels:
            if label not in manifest.coordinate_columns:
                continue
            fig = plot_heuristic_on_coordinates(manifest, h.name, label)
            path = save_figure(fig, output_dir / f"coords_{h.name}_{label}", fmt=fmt)
            click.echo(f"Saved: {path}")

    click.echo(f"All figures written to {output_dir}")


@main.command()
@click.argument("inspection_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None, help="Output video path.")
@click.option("--codec", default="h264", help="Video codec (h264, mpeg4, hevc_nvenc, av1_qsv).")
@click.option("--heuristic", default=None, help="Heuristic to highlight in video.")
def video(inspection_dir: Path, output: Path | None, codec: str, heuristic: str | None):
    """Generate an inspection video with matplotlib overlays."""
    from bikipy_inspect.manifest import InspectionManifest
    from bikipy_inspect.video import make_inspection_video

    manifest = InspectionManifest.from_directory(inspection_dir)
    make_inspection_video(manifest, output_path=output, codec=codec, heuristic_name=heuristic)
    click.echo("Inspection video created.")


@main.command()
@click.argument("inspection_dir", type=click.Path(exists=True, path_type=Path))
def info(inspection_dir: Path):
    """Display summary information about an inspection output."""
    from bikipy_inspect.manifest import InspectionManifest

    manifest = InspectionManifest.from_directory(inspection_dir)

    click.echo(f"Video: {manifest.video.path or 'N/A'}")
    click.echo(f"  FPS: {manifest.video.fps}, Frames: {manifest.video.total_frames}")
    click.echo(f"  Resolution: {manifest.video.resolution}")
    click.echo(f"  Duration: {manifest.video.duration_seconds:.1f}s")
    click.echo(f"  Meters/pixel: {manifest.settings.meters_per_pixel}")
    click.echo(f"Labels ({len(manifest.labels)}): {', '.join(manifest.labels) or 'N/A'}")
    click.echo(f"Perimeters ({len(manifest.perimeters)}):")
    for p in manifest.perimeters:
        click.echo(f"  - {p.label} ({p.shape})")
    click.echo(f"Heuristics ({len(manifest.heuristics)}):")
    for h in manifest.heuristics:
        click.echo(f"  - {h.name}: {h.seconds:.1f}s ({h.percentage:.1f}%)")
    click.echo(f"DataFrame: {manifest.df.shape[0]} rows x {manifest.df.shape[1]} columns")
