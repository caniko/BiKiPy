from typing import Optional

import click
from pydantic import DirectoryPath, validate_arguments

from bikipy.behaviour.mapping import experiment_name_to_class
from bikipy.cli import cli_root
from bikipy.ingress.utils.settings import auto_define_ingress_object
from bikipy.ingress.workflow.base import analyze_and_save, init_settings
from bikipy.reader.utils import merge_timestamps_with_dlc
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
    "project_root_directory",
    help="Path to the sequence formatted project directory, uses current directory on omition",
)
@click.option("-s", "--data_file_suffi  x", "framewise_coordinates_file_suffix", default=".h5")
@click.option("-d", "--dry_run", is_flag=True)
@validate_arguments
def init(
    ingress_method: str,
    experiment_name: str,
    project_root_directory: Optional[DirectoryPath] = None,
    framewise_coordinates_file_suffix: str = "h5",
    dry_run: bool = False,
) -> None:
    project_root_directory = current_path_or_arg_path(project_root_directory)

    if (project_root_directory / "settings.yaml").exists() and input(
        "Project has already been initialised, overwrite settings? y/N "
    ).strip().lower() != "y":
        return print("User aborted re-initialisation")

    init_settings(ingress_method, experiment_name, project_root_directory, framewise_coordinates_file_suffix, dry_run)


@ingress.command()
@click.argument("project_root_directory")
@validate_arguments
def analyze(project_root_directory: Optional[DirectoryPath]) -> None:
    analyze_and_save(current_path_or_arg_path(project_root_directory))


@ingress.command()
@click.option("-p", "project_root_directory")
@click.option("-x", "--delete_outdated", help="Outdated field will be removed", is_flag=True)
@click.option("-d", "--dry_run", is_flag=True)
@validate_arguments
def update(
    project_root_directory: Optional[DirectoryPath], delete_outdated: bool = False, dry_run: bool = False
) -> None:
    auto_define_ingress_object(current_path_or_arg_path(project_root_directory)).update_settings(
        delete_outdated=delete_outdated, dry_run=dry_run
    )


@ingress.command()
@click.option("-p", "project_root_directory")
@click.option("-o", "override_pattern")
@validate_arguments
def purge_cache(project_root_directory: Optional[DirectoryPath], override_pattern: Optional[str] = None) -> None:
    auto_define_ingress_object(current_path_or_arg_path(project_root_directory)).purge_cached_reads()


@ingress.command()
@click.option("-p", "project_root_directory")
@validate_arguments
def merge_coords_bonsai_timestamps(project_root_directory: Optional[DirectoryPath]) -> None:
    ingress = auto_define_ingress_object(current_path_or_arg_path(project_root_directory))
    merge_timestamps_with_dlc(ingress.dataset_directory_path, coordinate_file_lookup_expression="*.h5")
