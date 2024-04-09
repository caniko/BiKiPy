from typing import Optional

import click
from pydantic import DirectoryPath, validate_call

from bikipy.behaviour.mapping import experiment_name_to_class
from bikipy.cli import cli_root
from bikipy.reader.compute import merge_timestamps_with_dlc
from bikipy.utils.misc import current_path_or_arg_path


@cli_root.group
def ingress():
    pass


@ingress.command()
@click.argument("ingress_method")
@click.option(
    "-e", "--experiment_name", help=f"Name of experiment. Choose from:\n{', '.join(experiment_name_to_class.keys())}"
)
@click.option(
    "-p",
    "--path",
    "project_directory",
    help="Path to the sequence formatted project directory, uses current directory on omition",
)
@click.option("-d", "--dry_run", is_flag=True)
@validate_call
def init(
    ingress_method: str,
    experiment_name: str,
    project_directory: Optional[DirectoryPath] = None,
    dry_run: bool = False,
) -> None:
    project_directory = current_path_or_arg_path(project_directory)

    if (project_directory / "settings.yaml").exists() and input(
        "Project has already been initialised, overwrite settings? y/N "
    ).strip().lower() != "y":
        return print("User aborted re-initialisation")

    init_settings(ingress_method, experiment_name, project_directory, dry_run)


@ingress.command()
@click.argument("project_directory")
@validate_call
def analyze(project_directory: Optional[DirectoryPath]) -> None:
    analyze_and_save(current_path_or_arg_path(project_directory))


@ingress.command()
@click.option("-p", "project_directory")
@click.option("-x", "--delete_outdated", help="Outdated field will be removed", is_flag=True)
@click.option("-d", "--dry_run", is_flag=True)
@validate_call
def update(project_directory: Optional[DirectoryPath], delete_outdated: bool = False, dry_run: bool = False) -> None:
    update_settings(project_directory, delete_outdated=delete_outdated)


@ingress.command()
@click.option("-p", "project_directory")
@click.option("-o", "override_pattern")
@validate_call
def purge_cache(project_directory: Optional[DirectoryPath], override_pattern: Optional[str] = None) -> None:
    auto_define_ingress_object(current_path_or_arg_path(project_directory)).purge_cached_reads()


@ingress.command()
@click.option("-p", "project_directory")
@validate_call
def merge_coords_bonsai_timestamps(project_directory: Optional[DirectoryPath]) -> None:
    ingress = auto_define_ingress_object(current_path_or_arg_path(project_directory))
    merge_timestamps_with_dlc(ingress.dataset_directory, coordinate_file_lookup_expression="*.h5")
