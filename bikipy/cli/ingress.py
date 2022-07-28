import os
from pathlib import Path
from typing import Optional

import click
from pydantic import DirectoryPath, validate_arguments

from bikipy.cli import cli_root
from bikipy.behaviour.mapping import EXPERIMENT_NAME_TO_CLASS
from bikipy.ingress.core import (
    analyze_and_save,
    auto_define_ingress_object,
    init_settings,
)


@cli_root.group
def ingress():
    pass


@ingress.command()
@click.argument("ingress_method")
@click.option(
    "-e", "--experiment_name", help=f"Name of experiment. Choose from:\n{', '.join(EXPERIMENT_NAME_TO_CLASS.keys())}"
)
@click.option(
    "-p",
    "--path",
    "project_root_directory",
    help="Path to the sequence formatted project directory, uses current directory on omition",
)
@click.option("-s", "--data_file_suffix", "framewise_coordinates_file_suffix", default=".h5")
@click.option("-d", "--dry_run", is_flag=True)
@validate_arguments
def init(
    ingress_method: str,
    experiment_name: str,
    project_root_directory: Optional[DirectoryPath] = None,
    framewise_coordinates_file_suffix: str = "h5",
    dry_run: bool = False,
) -> None:
    project_root_directory = _define_project_root_directory(project_root_directory)

    if (project_root_directory / "settings.yaml").exists() and input(
        "Project has already been initialised, overwrite settings? y/N "
    ).strip().lower() != "y":
        return print("User aborted re-initialisation")

    init_settings(ingress_method, experiment_name, project_root_directory, framewise_coordinates_file_suffix, dry_run)


@ingress.command()
@click.argument("project_root_directory")
@validate_arguments
def analyze(project_root_directory: Optional[DirectoryPath]) -> None:
    analyze_and_save(_define_project_root_directory(project_root_directory))


@ingress.command()
@click.argument("project_root_directory")
@click.option("-x", "--delete_outdated", help="Outdated field will be removed", is_flag=True)
@click.option("-d", "--dry_run", is_flag=True)
@validate_arguments
def update(
    project_root_directory: Optional[DirectoryPath], delete_outdated: bool = False, dry_run: bool = False
) -> None:
    auto_define_ingress_object(_define_project_root_directory(project_root_directory)).update_settings(
        delete_outdated=delete_outdated, dry_run=dry_run
    )


@ingress.command()
@click.argument("project_root_directory")
@validate_arguments
def verify(project_root_directory: Optional[DirectoryPath]) -> None:
    auto_define_ingress_object(_define_project_root_directory(project_root_directory)).verify_project_structure()


def _define_project_root_directory(project_root_directory: Optional[DirectoryPath]):
    return project_root_directory or Path(os.curdir)
