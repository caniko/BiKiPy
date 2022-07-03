from logging import getLogger
from typing import ClassVar

from pydantic import PositiveInt

from bikipy.ingress.core import BaseIngress


logger = getLogger(__name__)


class PhaseIngress(BaseIngress):
    ingress_method: ClassVar[str] = "phase"

    def _dataset_reader(self) -> None:
        for phase_dir in self.dataset_directory_path.iterdir():
            phase_id = self._get_id_from_path_stem(phase_dir)

            for trial_kinematic_data_file_path in phase_dir.glob(f"*{self.kinematic_data_file_extension}"):
                phase_designated_trial_id = self._get_id_from_path_stem(trial_kinematic_data_file_path)
                trial_id = _define_trial_id(phase_id, phase_designated_trial_id)
                stage_index = self.metadata[trial_id]["stage"]

                self._trial_id_to_trial_class_name[trial_id] = self._trial_class_from_stage_index(trial_id)
                self._trial_id_to_keyword_arguments[trial_id] = {
                    "label": trial_id,
                    "animal_id": self.metadata[trial_id]["AnimalId"],
                    "stage": stage_index,
                    "coordinate_data_path": trial_kinematic_data_file_path,
                    **self._trial_id_to_keyword_arguments[trial_id],
                    **self._trialwise_plugins_for_trial_id(trial_id, phase_dir),
                }


def _define_trial_id(phase_id: str | PositiveInt, stage_index: str | PositiveInt):
    return f"{phase_id}_{stage_index}"
