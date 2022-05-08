from typing import Optional

import click
from pydantic import DirectoryPath

from bikipy.behaviour.mapping import NAME_TO_CLASS
from bikipy.ingress.sequence import sequence_generate_configuration


@click.group
def cli_root():
    pass


@cli_root.command()
@click.option("-e", "--experiment_name", help=f"Name of experiment. Choose from:\n{', '.join(NAME_TO_CLASS.keys())}")
@click.option(
    "-p",
    "--path",
    "root_directory",
    help="Path to the sequence formatted project directory, uses current directory on omition",
)
@click.option("-r", "--meter_pixel_ratio", help="Float/Decimal representation of the meter to pixel ratio")
@click.option("-s", "--data_file_suffix", "kinematic_data_file_extension", default=".h5")
@click.option("-a", "animals_have_plural_trial_sets", is_flag=True)
@click.option("-d", "--dry_run", is_flag=True)
def sequence(
    experiment_name: str,
    root_directory: Optional[DirectoryPath] = None,
    meter_pixel_ratio: Optional[float] = None,
    kinematic_data_file_extension: str = "h5",
    animals_have_plural_trial_sets: bool = False,
    dry_run: bool = False,
):
    return sequence_generate_configuration(
        root_directory=root_directory,
        experiment_name=experiment_name,
        meter_pixel_ratio=meter_pixel_ratio,
        kinematic_data_file_extension=kinematic_data_file_extension,
        animals_have_plural_trial_sets=animals_have_plural_trial_sets,
        dry_run=dry_run,
    )
