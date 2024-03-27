from logging import getLogger
from typing import ClassVar

from pydantic import DirectoryPath

from bikipy.core.typing import Label
from bikipy.ingress.plugin.core.plugin_scope import PluginScope
from bikipy.ingress.utils.io import analysis_cache_file_name_from_trial_id
from bikipy.ingress.workflow.base import BaseIngressWorkflow

logger = getLogger(__name__)


class AnimalIngressWorkflow(BaseIngressWorkflow):
    ingress_method: ClassVar[str] = "animal"

    def _dataset_reader(self) -> None:
        for animal_dir in self.dataset_directory.iterdir():
            if animal_dir.name.startswith(".") or animal_dir.is_file():
                continue

            animal_id = self._get_id_from_path_stem(animal_dir)

            for framewise_coordinates_path in self._coordinate_files_in_directory(animal_dir):
                stage_index = int(self._get_id_from_path_stem(framewise_coordinates_path).split(".")[0])

                trial_id = f"{animal_id}_{stage_index}"

                if self._to_skip_trial_id(trial_id):
                    continue

                plugin_data = {}
                for plugin_model in self._trialwise_plugins:
                    plugin_data_files = tuple(animal_dir.glob(f"{stage_index}.{plugin_model.code_key}*"))
                    if len(plugin_data_files) > 1:
                        msg = (
                            f"Only one file per trial: Animal {animal_id} -> Stage {stage_index} "
                            f"-> Plugin {plugin_model.human_readable_index}:\n{plugin_data_files}"
                        )
                        raise ValueError(msg)

                    try:
                        data_object = self._define_plugin(
                            plugin_model,
                            PluginScope.TRIALWISE,
                            data_path=plugin_data_files[0],
                            **self.get_plugin_config(plugin_model),
                        ).trialwise_and_metadata(trial_id=trial_id)
                    except IndexError:
                        continue

                    plugin_data[plugin_model.default_trial_argument_key or data_object.default_trial_argument_key] = (
                        data_object
                    )

                if self.experiment_class.has_stages:
                    self.trial_id_to_trial_class_name[trial_id] = self._trial_class_from_stage_index(
                        stage_index
                    ).__name__

                self.trial_id_to_keyword_arguments[trial_id] = {
                    "label": trial_id,
                    "animal_id": animal_id,
                    "framewise_coordinates_path": framewise_coordinates_path,
                    "analysis_series_cache_file_path": self.cache_directory_path
                    / analysis_cache_file_name_from_trial_id(trial_id),
                    **self.trialwise_plugins_for_trial_id(trial_id, animal_dir),
                    **self.trial_id_to_keyword_arguments[trial_id],
                    **plugin_data,
                }

            if self.only_one_instance_of_trial_class:
                break

    def trialwise_plugins_for_trial_id(self, trial_id: Label, trial_directory: DirectoryPath):
        return self._trialwise_plugins_for_trial_id(trial_id, trial_directory, "{trial_id}.{plugin_code_key}*")
