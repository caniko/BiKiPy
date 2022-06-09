import os
from typing import Optional

import click
from pydantic import DirectoryPath, validate_arguments

from bikipy.behaviour.mapping import EXPERIMENT_NAME_TO_CLASS
from bikipy.ingress.mapping import INGRESS_METHOD_NAME_TO_INIT_FUNC


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
    "root_directory",
    help="Path to the sequence formatted project directory, uses current directory on omition",
)
@click.option("-s", "--data_file_suffix", "kinematic_data_file_extension", default=".h5")
@click.option("-a", "animals_have_plural_trial_sets", is_flag=True)
@click.option("-d", "--dry_run", is_flag=True)
@validate_arguments
def init(
    ingress_method: str,
    experiment_name: str,
    root_directory: Optional[DirectoryPath] = None,
    kinematic_data_file_extension: str = "h5",
    animals_have_plural_trial_sets: bool = False,
    dry_run: bool = False,
) -> None:
    root_directory = root_directory or os.curdir
    if (root_directory / "settings.yaml").exists() and input(
        "Project has already been initialised, overwrite settings? y/N "
    ).strip().lower() != "y":
        return print("User aborted re-initialisation")

    INGRESS_METHOD_NAME_TO_INIT_FUNC[ingress_method.lower()](
        root_directory=root_directory,
        experiment_name=experiment_name,
        kinematic_data_file_extension=kinematic_data_file_extension,
        animals_have_plural_trial_sets=animals_have_plural_trial_sets,
        dry_run=dry_run,
    )
