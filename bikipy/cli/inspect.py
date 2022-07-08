from typing import Optional

import click
from pydantic import FilePath, DirectoryPath, validate_arguments

from bikipy.cli import cli_root
from bikipy.ingress.plugin.perimeter.single import inspect_annotations


@cli_root.group
def inspect():
    pass


@inspect.command
@click.argument("annotation_path")
@click.option(
    "-i",
    "--image",
    "image_directory",
    help="Path to the directory where the images coupled to the image names in the annotation files are defined",
)
@validate_arguments
def check_annotation(annotation_path: FilePath, image_directory: Optional[DirectoryPath] = None) -> None:
    inspect_annotations(annotation_path, image_directory)


@inspect.command
@click.argument("annotation_dir")
@validate_arguments
def check_annotations(annotation_dir: DirectoryPath = "") -> None:
    for annotation_file in annotation_dir.glob("perimeter-*"):
        inspect_annotations(annotation_file)
