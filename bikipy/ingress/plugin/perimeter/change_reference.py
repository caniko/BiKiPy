from functools import cached_property
from typing import ClassVar

from pydantic_numpy.dtype import NDArrayFp64

from bikipy.core.typing import Label
from bikipy.ingress.name_parser import PluginFileStemParse
from bikipy.ingress.plugin.core.base import BasePluginFile
from bikipy.ingress.plugin.core.mixins import IngressRequiredMixin
from bikipy.perimeter.base import BasePerimeter, Perimeter
from bikipy.utils.collection_utils import get_first_value_in_dict
from bikipy.utils.makesense import (
    get_point_from_makesense_row,
    image_name_to_point_from_makesense,
)


class ChangeReferencePluginFileStemParse(PluginFileStemParse):
    def __pop_split_till_empty__(self) -> None:
        self.label = self.split.popleft()

        try:
            self._new_label = self.split.popleft()
        except IndexError:
            self._new_label = None

    @property
    def new_label(self):
        if not self._new_label:
            msg = f"The reference file has no new_label defined, hence it cannot be used globally: {self.stem}"
            raise AttributeError(msg)
        return self._new_label


class PluginChangeReference(BasePluginFile, IngressRequiredMixin):
    ingress_key = "change_reference"
    code_key = "change_reference"
    default_trial_argument_key = "change_reference"

    plugin_file_stem_parser = ChangeReferencePluginFileStemParse

    human_readable_index = "ChangeReference"
    name_human_readable_index: ClassVar[str] = "ChangeReferenceImageName"

    @cached_property
    def image_name_to_re_referencing_point(self) -> dict[str, NDArrayFp64]:
        return image_name_to_point_from_makesense(self.data_path, only_point=False)

    def trialwise_and_metadata(self, trial_id: Label, naive: bool = False) -> Perimeter | dict:
        self._assert_correct_scope_trialwise_metadata()

        trial_id_image_name = self.ingress.metadata.loc[trial_id, self.name_human_readable_index]
        reference_data = self.image_name_to_re_referencing_point[trial_id_image_name]

        original_perimeter = self.ingress.ingress_defined_perimeters[self.stem_info.label]

        if isinstance(original_perimeter, dict):
            # Grouped with PerimeterSet.group()
            result = {}
            for group_name, perimeters in original_perimeter.items():
                if isinstance(perimeters, BasePerimeter):
                    perimeters = perimeters.change_reference(
                        new_reference=get_point_from_makesense_row(reference_data),
                        makesense_image_name=reference_data["image_name"],
                    )
                else:
                    perimeters = tuple(
                        perimeter.change_reference(
                            new_reference=get_point_from_makesense_row(reference_data),
                            makesense_image_name=reference_data["image_name"],
                        )
                        for perimeter in perimeters
                    )
                result[group_name] = perimeters
        else:
            result = self.ingress.ingress_defined_perimeters[self.original_label].change_reference(
                new_reference=get_point_from_makesense_row(reference_data),
                makesense_image_name=reference_data["image_name"],
            )

        return result

    @property
    def globally_defined(self) -> Perimeter:
        self._assert_correct_scope_global()
        assert (
            len(self.image_name_to_re_referencing_point) == 1
        ), "There can only be one re-reference point in a globally_defined reference"

        reference_data = get_first_value_in_dict(self.image_name_to_re_referencing_point)

        result = self.ingress.ingress_defined_perimeters[self.original_label].globally_defined.change_reference(
            get_point_from_makesense_row(reference_data), makesense_image_name=reference_data["image_name"]
        )
        result.label = self.new_label

        return result
