from logging import getLogger

from projectkit.model.cds import CdsHierarchy, CdsHomologs, CdsSingle
from projectkit.model.config.jit import ProjectKitJITConfiguration
from pydantic import DirectoryPath

from bikipy import BikipyRuntimeSettings
from bikipy.behaviour.core.enclosure.base import EnclosedExperiment, EnclosedTrial
from bikipy.behaviour.radial_arm import BaseRadialMazeExperiment
from bikipy.ingress.plugin import (
    PluginRadial,
    PluginSinglePerimeter,
)


logger = getLogger(__name__)


class ProjectKitJITBikipyConfiguration(ProjectKitJITConfiguration):
    project_name = "bikipy"

    @staticmethod
    def jit_init(*, ingress_method: str, experiment_name: str, project_directory: DirectoryPath) -> dict:
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
