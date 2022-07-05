from functools import cached_property
from typing import ClassVar

from pydantic import PositiveInt

from bikipy.core.typing import NDArrayFp64
from bikipy.ingress.plugin.base import BasePluginFile
from bikipy.perimeter.base import Perimeter
from bikipy.utils.collection_utils import get_first_value_in_dict
from bikipy.utils.io.makesense import image_name_to_point_from_makesense, get_point_from_makesense_row


class PluginChangeReference(BasePluginFile):
    ingress_key = "change_reference_definition_strategy"
    code_key = "change_reference"
    bikipy_trial_key = "change_reference"

    human_readable_index = "ChangeReference"
    _human_readable_index_image_name: ClassVar[str] = "ChangeReferenceImageName"

    @property
    def label(self) -> str:
        return self._info[1]

    @property
    def original_label(self) -> str:
        return self.label

    @property
    def new_label(self) -> str:
        try:
            return self._info[2]
        except IndexError as e:
            msg = f"The reference file has no new_label defined, hence it cannot be used globally: {self.data_path}"
            raise AttributeError(msg) from e

    @cached_property
    def image_name_to_re_referencing_point(self) -> dict[str, NDArrayFp64]:
        return image_name_to_point_from_makesense(self.data_path, only_point=False)

    def trialwise_and_metadata(self, trial_id: str | PositiveInt) -> Perimeter:
        trial_id_image_name = self.ingress.metadata.loc[trial_id, self._human_readable_index_image_name]
        reference_data = self.image_name_to_re_referencing_point[trial_id_image_name]

        return self.ingress.ingress_defined_perimeters[self.original_label].globally_defined.change_reference(
            get_point_from_makesense_row(reference_data), makesense_image_name=reference_data["image_name"]
        )

    @property
    def globally_defined(self) -> Perimeter:
        assert (
            len(self.image_name_to_re_referencing_point) == 1
        ), "There can only be one re-reference point in a globally_defined reference"

        reference_data = get_first_value_in_dict(self.image_name_to_re_referencing_point)

        result = self.ingress.ingress_defined_perimeters[self.original_label].globally_defined.change_reference(
            get_point_from_makesense_row(reference_data), makesense_image_name=reference_data["image_name"]
        )
        result.label = self.new_label

        return result
