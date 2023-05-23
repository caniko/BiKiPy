from logging import getLogger
from typing import Optional

from projectkit.model.jit import ProjectKitJITConfiguration
from projectkit.utils.misc import here_or_there
from pydantic import DirectoryPath
from schemantic.model.schema import GroupSchema, SingleSchema

from bikipy import BikipyRuntimeSettings
from bikipy._constant import (
    ENCLOSURE_MAP_NAME,
    EXPERIMENT_MAP_NAME,
    INGRESS_MAP_NAME,
    PERIMETER_MAP_NAME,
    PHYSICAL_OBJECT_MAP_NAME,
    PLUGIN_MAP_NAME,
    PROJECTKIT_CONFIG_KEY_ORDER,
    READER_MAP_NAME,
    RUNTIME_SETTINGS_MAP_NAME,
    TRIAL_MAP_NAME,
)
from bikipy.behaviour.core.enclosure.base import EnclosedExperiment, EnclosedTrial
from bikipy.behaviour.mapping import experiment_name_to_class
from bikipy.behaviour.radial_arm.base import BaseRadialMazeExperiment
from bikipy.feature.qualia.physical_object.heuristic.mapping import HEURISTIC_MAP
from bikipy.ingress.plugin.perimeter.enclosure import PluginEnclosure
from bikipy.ingress.workflow.animal import AnimalIngressWorkflow
from bikipy.ingress.workflow.animal_day import AnimalDayIngressWorkflow
from bikipy.ingress.workflow.phase import PhaseIngressWorkflow
from bikipy.reader import DeepLabCutReader

logger = getLogger(__name__)

INGRESS_METHOD_NAME_TO_INGRESS_CLASS = {
    "animal": AnimalIngressWorkflow.__name__,
    "animal_day": AnimalDayIngressWorkflow.__name__,
    "phase": PhaseIngressWorkflow.__name__,
}


class ProjectKitJITBikipyConfiguration(ProjectKitJITConfiguration):
    config_key_order = PROJECTKIT_CONFIG_KEY_ORDER

    project_name = "bikipy"

    root_class_config_key = INGRESS_MAP_NAME
    root_class_name_to_class = {
        cls.__name__: cls for cls in (AnimalIngressWorkflow, AnimalDayIngressWorkflow, PhaseIngressWorkflow)
    }

    def jit_init(
        self,
        ingress_method: str,
        experiment_name: str,
        project_directory: Optional[DirectoryPath] = None,
        qualia_heuristic: Optional[list[str]] = None,
    ) -> dict:
        from bikipy.ingress.plugin.perimeter.radial_maze import PluginRadial
        from bikipy.ingress.plugin.perimeter.single import PluginSinglePerimeter

        try:
            ingress_method = INGRESS_METHOD_NAME_TO_INGRESS_CLASS[ingress_method]
        except KeyError:
            pass

        experiment_name = experiment_name.lower()

        self.root_class_name = ingress_method
        self.root_class_init_kwargs["experiment_class_name"] = experiment_name

        project_directory = here_or_there(project_directory)

        logger.info(f"Generating experiment configuration at {project_directory}")

        experiment_class = experiment_name_to_class[experiment_name]
        cds_single = [
            SingleSchema(
                manual_mapping_name=self.root_class_config_key,
                model=self.root_class_name_to_class[ingress_method],
            ),
            SingleSchema(manual_mapping_name=RUNTIME_SETTINGS_MAP_NAME, model=BikipyRuntimeSettings),
            SingleSchema(
                manual_mapping_name=READER_MAP_NAME, model=DeepLabCutReader
            ),  # TODO: Cleo option to change reader
            SingleSchema(manual_mapping_name=EXPERIMENT_MAP_NAME, model=experiment_class),
        ]
        cds_homologs = []
        cds_hierarchical = []

        plugin_models = set()

        if len(experiment_class.trial_classes) == 1:
            cds_single.append(
                SingleSchema(manual_mapping_name=TRIAL_MAP_NAME, model=experiment_class.trial_classes.pop())
            )
        else:
            cds_hierarchical.append(
                GroupSchema.from_models(mapping_name=TRIAL_MAP_NAME, models=experiment_class.trial_classes)
            )

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
                cds_single.append(
                    SingleSchema(
                        manual_mapping_name=ENCLOSURE_MAP_NAME,
                        model=trial_class_to_perimeter_enclosure.pop(tuple(trial_class_to_perimeter_enclosure)[0]),
                    )
                )
            else:
                cds_hierarchical.append(
                    GroupSchema.from_models(
                        mapping_name=ENCLOSURE_MAP_NAME,
                        instance_names=set({c.__name__ for c in trial_class_to_perimeter_enclosure}),
                        models=set(trial_class_to_perimeter_enclosure.values()),
                    )
                )
            plugin_models.add(PluginEnclosure)

        if experiment_class.at_least_one_trial_has_perimeter:
            cds_hierarchical.append(
                GroupSchema.from_models(
                    mapping_name=PERIMETER_MAP_NAME,
                    models=experiment_class.trial_perimeter_label_to_perimeter_class,
                )
            )
            plugin_models.add(PluginSinglePerimeter)
            if issubclass(experiment_class, BaseRadialMazeExperiment):
                plugin_models.add(PluginRadial)

            if experiment_class.at_least_one_trial_has_physical_object:
                assert qualia_heuristic, "Experiments with physical objects require qualia profiling"
                models = set()
                for heuristic in qualia_heuristic:
                    try:
                        models.add(HEURISTIC_MAP[heuristic])
                    except KeyError:
                        msg = (
                            f"The defined heuristic key, {heuristic}, is not defined. "
                            f"Choose from the following: {', '.join(tuple(HEURISTIC_MAP))}"
                        )
                        raise KeyError(msg)

                cds_hierarchical.append(GroupSchema.from_models(models=models, mapping_name=PHYSICAL_OBJECT_MAP_NAME))

        if plugin_models:
            cds_hierarchical.append(GroupSchema.from_models(mapping_name=PLUGIN_MAP_NAME, models=plugin_models))

        return {
            "project_directory": project_directory,
            "cds_singles": set(cds_single),
            "cds_homologs": set(cds_homologs),
            "cds_groups": set(cds_hierarchical),
        }
