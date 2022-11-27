from logging import getLogger
from typing import ClassVar

from pydantic import DirectoryPath

from bikipy.core.typing import TrialId
from bikipy.ingress.workflow.base import BaseIngress

logger = getLogger(__name__)


class AnimalIngress(BaseIngress):
    ingress_method: ClassVar[str] = "animal"

    def _dataset_reader(self) -> None:
        def define_trial_id():
            return f"{animal_id}_{stage_index}"

        for animal_dir in self.dataset_directory_path.iterdir():
            animal_id = self._get_id_from_path_stem(animal_dir)

            for framewise_coordinates_path in self._glob_coordinate_files_in_directory(animal_dir):
                stage_index = int(self._get_id_from_path_stem(framewise_coordinates_path).split(".")[0])

                trial_id = define_trial_id()

                if trial_id not in self.metadata.index:
                    continue

                plugin_data = {}
                for plugin_model in self._trial_wise_plugins:
                    plugin_data_files = tuple(animal_dir.glob(f"{stage_index}.{plugin_model.code_key}*"))
                    if len(plugin_data_files) > 1:
                        msg = f"Only one file per trial: Animal {animal_id} -> Stage {stage_index} -> Plugin {plugin_model.human_readable_index}"
                        raise ValueError(msg)

                    try:
                        data_object = plugin_model(data_path=plugin_data_files[0], ingress=self).trialwise_and_metadata(
                            trial_id=trial_id
                        )
                    except IndexError:
                        continue

                    plugin_data[plugin_model.bikipy_trial_key or data_object.bikipy_trial_key] = data_object

                self._trial_id_to_trial_class_name[trial_id] = self._trial_class_from_stage_index(stage_index)
                self._trial_id_to_keyword_arguments[trial_id] = {
                    "label": trial_id,
                    "animal_id": animal_id,
                    "framewise_coordinates_path": framewise_coordinates_path,
                    **self.trialwise_plugins_for_trial_id(trial_id, animal_dir),
                    **self._trial_id_to_keyword_arguments[trial_id],
                    **plugin_data,
                }

    def trialwise_plugins_for_trial_id(self, trial_id: TrialId, trial_directory: DirectoryPath):
        return self._trialwise_plugins_for_trial_id(trial_id, trial_directory, "{trial_id}.{plugin_code_key}*")
