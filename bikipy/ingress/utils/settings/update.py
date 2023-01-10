import json
from typing import Optional

from pydantic import DirectoryPath, validate_arguments

from bikipy.ingress.utils import settings
from bikipy.ingress.utils.io import dump_settings, get_project_settings_path
from bikipy.ingress.utils.settings import (
    auto_define_ingress_object,
    get_definable_settings,
)
from bikipy.ingress.workflow.base import init_settings
from bikipy.utils.misc import current_path_or_arg_path


@validate_arguments
def update_settings(
    manual_project_directory_path: Optional[DirectoryPath] = None,
    delete_outdated: bool = False,
    dry_run: bool = False,
    silent: bool = False,
) -> None:
    project_path = current_path_or_arg_path(manual_project_directory_path)
    ingress = auto_define_ingress_object(
        project_path, deprecated_file_name=not get_project_settings_path(project_path).exists()
    )

    new_settings = init_settings(
        ingress.ingress_method,
        ingress.experiment_name,
        ingress.project_root_directory,
        ingress.framewise_coordinates_file_suffix,
        dry_run=True,
        silent=True,
    )

    kwargs = {"delete_outdated": delete_outdated}

    for key in ("ingress", "definition_strategies", "perimeter"):
        if key in ingress.settings:
            try:
                new_settings[key] = settings.update_dictionary(ingress.settings[key], new_settings[key], **kwargs)
            except KeyError:
                pass

    for key in ("manual_reader_kwargs", "experiment"):
        if key in ingress.settings:
            try:
                new_settings[key] = settings.update_defined_values(ingress.settings[key], new_settings[key], **kwargs)
            except KeyError:
                pass

    if "trial" in ingress.settings:
        new_settings["trial"]["common"] = settings.update_defined_values(
            ingress.settings["trial"]["common"], new_settings["trial"]["common"], **kwargs
        )
        common_settings_between_trials = get_definable_settings(new_settings["trial"]["common"])
        for trial_class_name, trial_class_settings in new_settings["trial"]["specific"].items():
            try:
                new_settings["trial"]["specific"][trial_class_name] = settings.update_defined_values(
                    ingress.settings["trial"]["specific"][trial_class_name],
                    trial_class_settings,
                    common_settings=common_settings_between_trials,
                    **kwargs,
                )
            except KeyError:
                pass

    for field, value in ingress.settings.items():
        if isinstance(value, dict):
            continue
        if field in new_settings and value:
            new_settings[field] = value

    if not dry_run:
        dump_settings(get_project_settings_path(project_path), new_settings)

    if not silent:
        print(json.dumps(new_settings, indent=2))
