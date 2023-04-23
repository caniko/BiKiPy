from logging import getLogger
from typing import Optional

from ordered_set import OrderedSet
from schemantic.model.schema import GroupSchema, HomologSchema, SingleSchema
from projectkit.model.config.jit import ProjectKitJITConfiguration
from projectkit.utils.misc import here_or_there
from pydantic import DirectoryPath

from bikipy import BikipyRuntimeSettings
from bikipy.behaviour.core.enclosure.base import EnclosedExperiment, EnclosedTrial
from bikipy.behaviour.mapping import experiment_name_to_class
from bikipy.behaviour.radial_arm import BaseRadialMazeExperiment
from bikipy.ingress.workflow import INGRESS_METHOD_NAME_TO_INGRESS_CLASS
from bikipy.reader import DeepLabCutReader

logger = getLogger(__name__)


class ProjectKitJITBikipyConfiguration(ProjectKitJITConfiguration):
    ingress_method: str = ...
    experiment_name: str = ...

    config_key_order = OrderedSet(
        ("manual", "ingress", "experiment", "trial", "enclosure", "perimeter", "perimeter_plugin", "runtime_settings")
    )

    project_name = "bikipy"

    root_class_config_key = "ingress"
    root_class_name_to_class = INGRESS_METHOD_NAME_TO_INGRESS_CLASS

    def jit_init(self, project_directory: Optional[DirectoryPath] = None) -> dict:
        from bikipy.ingress.plugin.perimeter.radial_maze import PluginRadial
        from bikipy.ingress.plugin.perimeter.single import PluginSinglePerimeter

        project_directory = here_or_there(project_directory)

        logger.info(f"Generating experiment configuration at {project_directory}")

        experiment_class = experiment_name_to_class[self.experiment_name.lower()]
        cds_single = [
            SingleSchema(
                manual_mapping_name=self.root_class_config_key, model=self.root_class_name_to_class[self.ingress_method]
            ),
            SingleSchema(manual_mapping_name="runtime_settings", model=BikipyRuntimeSettings),
            SingleSchema(manual_mapping_name="reader", model=DeepLabCutReader),  # TODO: Cleo option to change reader
            SingleSchema(manual_mapping_name="experiment", model=experiment_class),
        ]
        cds_homologs = []
        cds_hierarchical = [GroupSchema.from_models(mapping_name="trial", models=set(experiment_class.trial_classes))]

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
                    HomologSchema(
                        manual_mapping_name="enclosure",
                        instance_names=set({c.__name__ for c in experiment_class.trial_class_names}),
                        model_schema=trial_class_to_perimeter_enclosure.pop(
                            tuple(trial_class_to_perimeter_enclosure)[0]
                        ),
                    )
                )
            else:
                cds_hierarchical.append(
                    GroupSchema.from_models(
                        mapping_name="enclosure",
                        instance_names=set({c.__name__ for c in trial_class_to_perimeter_enclosure}),
                        models=set(trial_class_to_perimeter_enclosure.values()),
                    )
                )

        if experiment_class.at_least_one_trial_has_perimeter:
            cds_hierarchical.append(
                GroupSchema.from_models(
                    mapping_name="perimeter",
                    models=experiment_class.trial_perimeter_label_to_perimeter_class,
                )
            )
            perimeter_plugin_classes = {PluginSinglePerimeter}
            if issubclass(experiment_class, BaseRadialMazeExperiment):
                perimeter_plugin_classes.add(PluginRadial)
            cds_hierarchical.append(
                GroupSchema.from_models(
                    mapping_name="perimeter_plugin",
                    models=set(perimeter_plugin_classes),
                )
            )

        return {
            "project_directory": project_directory,
            "cds_singles": set(cds_single),
            "cds_homologs": set(cds_homologs),
            "cds_groups": set(cds_hierarchical),
        }
