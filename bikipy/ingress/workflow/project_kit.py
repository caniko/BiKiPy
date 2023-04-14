from pathlib import Path

from projectkit.config import ProjectKitConfiguration
from projectkit.cds_argument import CdsArgument
from projectkit.pre_made import current_directory
from pydantic import DirectoryPath, validate_arguments


@validate_arguments
def find_components(components_root_directory: DirectoryPath) -> list[str, ...]:
    return [item.stem for item in components_root_directory.iterdir() if item.is_dir()]


projectkit_configuration = ProjectKitConfiguration(
    project_name="mask_to_mesh",
    field_map_to_callable={("World", "defined", "working_directory"): current_directory},
    cds_single_instance_interface={Ingress},
    cds_homolog_hierarchical_interface=(
        (
            Component,
            find_components,
            frozenset(
                {
                    CdsArgument(
                        arg_name="components_root_directory",
                        abbreviation="c",
                        map_to_config=("World", "defined", "data_directory"),
                        type=Path,
                        required=True,
                    ),
                }
            ),
        ),
    ),
)

projectkit_cli = projectkit_configuration.cli_project_kit_click_group(cli_group_name="mask2mesh-cli")
