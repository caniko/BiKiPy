import os
import pstats
from abc import ABC, abstractmethod
from cProfile import Profile
from functools import cached_property
from itertools import chain
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Optional, TypeVar

import numpy as np
import pandas as pd
from inflection import underscore
from projectkit.model.cds import CdsHierarchy, CdsHomologs, CdsSingle
from projectkit.model.config.jit import JITProjectKitConfiguration
from projectkit.model.project import BaseProjectKitModel
from pydantic import DirectoryPath, FilePath, PositiveInt, validate_arguments
from pydantic_numpy.dtype import NDArrayFp64

from bikipy import set_bikipy_settings_from_dict, BikipyRuntimeSettings
from bikipy.behaviour.core.enclosure.base import EnclosedExperiment, EnclosedTrial
from bikipy.behaviour.radial_arm import BaseRadialMazeExperiment
from bikipy.core.typing import Label
from bikipy.ingress.plugin import (
    PluginMeterPerPixel,
    ingress_key_to_model,
    PluginRadial,
    PluginSinglePerimeter,
)
from bikipy.ingress.plugin.base import Plugin, PluginScope
from bikipy.ingress.plugin.meters_per_pixel import (
    detect_meters_per_pixel_in_perimeter_directory,
)

from bikipy.ingress.utils.io import (
    get_inspect_directory_path,
    get_plugin_directory_path,
    get_project_settings_path,
    infer_metadata_path,
    load_settings,
    result_directory_path,
)
from bikipy.perimeter.base import Perimeter
from bikipy.reader.base import BaseReader
from bikipy.utils.collection_utils import (
    copycat_assumes_levels_of_icon,
    get_first_value_in_dict,
)
from bikipy.utils.misc import defaultdict_dict_factory, sheet_names_from_path

logger = getLogger(__name__)


class ProjectKitJITBikipyConfiguration(JITProjectKitConfiguration):
    project_name = "bikipy"

    def jit_init(self, ingress_method: str, experiment_name: str, project_directory: DirectoryPath) -> dict:
        from bikipy.ingress.workflow import INGRESS_METHOD_NAME_TO_INGRESS_CLASS
        from bikipy.behaviour.mapping import experiment_name_to_class

        logger.info(f"Generating experiment configuration at {project_directory}")

        experiment_class = experiment_name_to_class[experiment_name.lower()]
        cds_single = [
            CdsSingle(mapping_name="runtime_settings", cds_class=BikipyRuntimeSettings),
            CdsSingle(mapping_name="ingress", cds_class=INGRESS_METHOD_NAME_TO_INGRESS_CLASS[ingress_method]),
            CdsSingle(mapping_name="experiment", cds_class=experiment_class),
        ]
        cds_homologs = []
        cds_hierarchical = [CdsHierarchy(mapping_name="trials", cds_classes=frozenset(experiment_class.trial_classes))]

        if issubclass(experiment_class, EnclosedExperiment):
            trial_class_to_perimeter_enclosure = {
                trial_class: trial_class.trial_perimeter_enclosure_class
                for trial_class in experiment_class.trial_classes
                if issubclass(trial_class, EnclosedTrial)
            }
            assert (
                trial_class_to_perimeter_enclosure
            ), f"{experiment_class.__name__}, is an enclosed experiment, but has no class"
            if len(trial_class_to_perimeter_enclosure) == 1:
                cds_homologs.append(
                    CdsHomologs(
                        manual_mapping_name="enclosure",
                        instance_names=frozenset({c.__name__ for c in experiment_class.trial_class_names}),
                        cds_class=trial_class_to_perimeter_enclosure.pop(tuple(trial_class_to_perimeter_enclosure)[0]),
                    )
                )
            else:
                cds_hierarchical.append(
                    CdsHierarchy(
                        mapping_name="enclosure",
                        instance_names=frozenset({c.__name__ for c in trial_class_to_perimeter_enclosure}),
                        cds_classes=frozenset(trial_class_to_perimeter_enclosure.values()),
                    )
                )

        if experiment_class.at_least_one_trial_has_perimeter:
            cds_hierarchical.append(
                CdsHierarchy(
                    mapping_name="perimeters",
                    cds_classes=experiment_class.trial_perimeter_label_to_perimeter_class,
                )
            )
            perimeter_plugin_classes = {PluginSinglePerimeter}
            if issubclass(experiment_class, BaseRadialMazeExperiment):
                perimeter_plugin_classes.add(PluginRadial)
            cds_hierarchical.append(
                CdsHierarchy(
                    mapping_name="perimeter_plugin",
                    cds_classes=frozenset(perimeter_plugin_classes),
                )
            )

        return {
            "project_directory": project_directory,
            "cds_single_interface": frozenset(cds_single),
            "cds_homolog_interface": frozenset(cds_homologs),
            "cds_hierarchical_interface": frozenset(cds_hierarchical),
        }
