from logging import getLogger
from typing import ClassVar

from pydantic import DirectoryPath

from bikipy.core.typing import Label
from bikipy.ingress.plugin.core.plugin_scope import PluginScope
from bikipy.ingress.workflow.base import BaseIngressWorkflow

logger = getLogger(__name__)


class AnimalDayIngressWorkflow(BaseIngressWorkflow):
    ingress_method: ClassVar[str] = "animal_day"

    def _dataset_reader(self) -> None:
        def define_trial_id():
            return f"{animal_id}_{day}_{daily_trial_number}"

        for animal_dir in self.dataset_directory.iterdir():
            if animal_dir.is_file():
                continue

            animal_id = self._get_id_from_path_stem(animal_dir)

            for day_dir in animal_dir.iterdir():
                if day_dir.name.startswith("."):
                    continue

                day = day_dir.stem.lower().replace("D", "").strip()

                for framewise_coordinates_path in self._coordinate_files_in_directory(day_dir):
                    daily_trial_number = int(self._get_id_from_path_stem(framewise_coordinates_path).split(".")[0])

                    trial_id = define_trial_id()

                    if self._to_skip_trial_id(trial_id):
                        continue

                    plugin_data = {}
                    for plugin_model in self._trialwise_plugins:
                        plugin_data_files = tuple(day_dir.glob(f"{daily_trial_number}.{plugin_model.code_key}*"))
                        if len(plugin_data_files) > 1:
                            msg = f"Only one file per trial: Animal {animal_id} -> Stage {daily_trial_number} -> Plugin {plugin_model.human_readable_index}"
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

                        plugin_data[
                            plugin_model.default_trial_argument_key or data_object.default_trial_argument_key
                        ] = data_object

                    #  TODO: Must implement a versatile way of determining when singular trial_class
                    #        and many trial_classes across all ingress
                    self.trial_id_to_trial_class_name[trial_id] = self.experiment_class.trial_class

                    self.trial_id_to_keyword_arguments[trial_id] = {
                        "label": trial_id,
                        "animal_id": animal_id,
                        "framewise_coordinates_path": framewise_coordinates_path,
                        **self.trialwise_plugins_for_trial_id(trial_id, animal_dir),
                        **self.trial_id_to_keyword_arguments[trial_id],
                        **plugin_data,
                    }
                    if self._designator_id_to_kwargs:
                        self.trial_id_to_keyword_arguments[trial_id].update(
                            self._designator_id_to_kwargs[self._trial_id_to_designator_id[trial_id]]
                        )

    def trialwise_plugins_for_trial_id(self, trial_id: Label, trial_directory: DirectoryPath):
        return self._trialwise_plugins_for_trial_id(trial_id, trial_directory, "{trial_id}.{plugin_code_key}*")
