from logging import getLogger
from pathlib import Path
from typing import ClassVar

from pydantic import DirectoryPath, FilePath

from bikipy.ingress.core import BaseIngress


logger = getLogger(__name__)


class PhaseIngress(BaseIngress):
    ingress_method: ClassVar[str] = "phase"

    def _ingress_reader(self):
        for phase_dir in self.dataset_directory_path.iterdir():
            phase_id = self._get_id_from_path_stem(phase_dir)

            for trial_kinematic_data_file_path in phase_dir.glob(f"*{self.kinematic_data_file_extension}"):
                phase_designated_trial_id = self._get_id_from_path_stem(trial_kinematic_data_file_path)
                trial_id = _define_trial_id(phase_id, phase_designated_trial_id)
                stage_index = self.metadata[trial_id]["stage"]

                self._trial_id_to_trial_class_name[trial_id] = self._trial_class_from_stage_index(trial_id)

                trial_id_kwargs = {
                    "label": trial_id,
                    "animal_id": self.metadata[trial_id]["AnimalId"],
                    "stage": stage_index,
                    "coordinate_data_path": trial_kinematic_data_file_path,
                }
                for plugin_info in self._trial_wise_plugins:
                    plugin_data_files = tuple(phase_dir.glob(f"{phase_designated_trial_id}.{plugin_info['code_key']}*"))
                    if len(plugin_data_files) > 1:
                        msg = f"Plugin {plugin_info['human_readable_index']}: Only one file per trial"
                        raise ValueError(msg)

                    trial_id_kwargs[plugin_info["bikipy_trial_key"]] = plugin_info["file_path_to_value"](
                        plugin_data_files[0], trial_id, self
                    )

                self._trial_id_to_keyword_arguments[trial_id] = {
                    **self._trial_id_to_keyword_arguments[trial_id],
                    **trial_id_kwargs,
                }


def _define_trial_id(phase_id: str | int, stage_index: str | int):
    return f"{phase_id}_{stage_index}"
