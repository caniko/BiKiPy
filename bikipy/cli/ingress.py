import os
from pathlib import Path
from typing import Optional

import click
from pydantic import DirectoryPath, validate_arguments

from bikipy.behaviour.mapping import EXPERIMENT_NAME_TO_CLASS
from bikipy.ingress import INGRESS_METHOD_NAME_TO_INIT_FUNC
from bikipy.ingress.core import analyze_and_save, auto_define_ingress_object


@click.group
def cli_root():
    pass


@cli_root.command()
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
@click.option("-s", "--data_file_suffix", "kinematic_data_file_extension", default=".h5")
@click.option("-a", "animals_have_plural_trial_sets", is_flag=True)
@click.option("-d", "--dry_run", is_flag=True)
@validate_arguments
def init(
    ingress_method: str,
    experiment_name: str,
    project_root_directory: Optional[DirectoryPath] = None,
    kinematic_data_file_extension: str = "h5",
    animals_have_plural_trial_sets: bool = False,
    dry_run: bool = False,
) -> None:
    project_root_directory = _define_project_root_directory(project_root_directory)

    if (project_root_directory / "settings.yaml").exists() and input(
        "Project has already been initialised, overwrite settings? y/N "
    ).strip().lower() != "y":
        return print("User aborted re-initialisation")

    INGRESS_METHOD_NAME_TO_INIT_FUNC[ingress_method.lower()](
        project_root_directory=project_root_directory,
        experiment_name=experiment_name,
        kinematic_data_file_extension=kinematic_data_file_extension,
        animals_have_plural_trial_sets=animals_have_plural_trial_sets,
        dry_run=dry_run,
    )


@cli_root.command()
@click.argument("project_root_directory")
@validate_arguments
def analyze(project_root_directory: Optional[DirectoryPath] = None) -> None:
    analyze_and_save(_define_project_root_directory(project_root_directory))


@cli_root.command()
@click.argument("project_root_directory")
@click.option("-x", "--delete_outdated", help="Outdated field will be removed", is_flag=True)
@click.option("-d", "--dry_run", is_flag=True)
@validate_arguments
def update(
    project_root_directory: Optional[DirectoryPath] = None, delete_outdated: bool = False, dry_run: bool = False
) -> None:
    auto_define_ingress_object(_define_project_root_directory(project_root_directory)).update_settings(
        delete_outdated=delete_outdated, dry_run=dry_run
    )


@cli_root.command()
@click.argument("project_root_directory")
@validate_arguments
def verify(project_root_directory: Optional[DirectoryPath] = None) -> None:
    auto_define_ingress_object(_define_project_root_directory(project_root_directory)).verify_project_structure()


def _define_project_root_directory(project_root_directory: Optional[DirectoryPath] = None):
    return project_root_directory or Path(os.curdir)
