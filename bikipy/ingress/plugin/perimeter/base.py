from abc import ABC
from typing import ClassVar, Optional

from bikipy.core.typing import Label
from bikipy.ingress.plugin.core.base import BasePluginFile
from bikipy.ingress.plugin.core.mixins import HasReferenceMixin, IngressRequiredMixin
from bikipy.perimeter.base import (
    SinglePerimeter,
    StringPerimeterShapes,
    perimeter_set_from_makesense,
)


class AbstractPerimeterPlugin(BasePluginFile, HasReferenceMixin, IngressRequiredMixin, ABC):
    manual_shape: Optional[StringPerimeterShapes]

    warn_missing_re_reference_file: ClassVar[bool] = False

    plural_entries = True

    ingress_key = "perimeter"
    code_key = "perimeter"
    default_trial_argument_key = "label_to_perimeter"
    human_readable_index = "Perimeter"

    def perimeter_mapper(
        self, trial_id: Optional[Label] = None, **perimeter_model_field_kwargs
    ) -> dict[str, SinglePerimeter]:
        image_name_to_perimeter_set = perimeter_set_from_makesense(
            self.data_path,
            self.manual_shape or self.stem_info.shape,  # single -> stem_info.shape
            meters_per_pixel=self.ingress.get_meter_per_pixel(trial_id),
            reference_point_array=self.reference_point,
            inspection_fig_output_path=self.ingress.inspect_directory_path,
            **perimeter_model_field_kwargs,
        )

        result = {}
        for image_name, perimeter_set in image_name_to_perimeter_set.items():
            for label, perimeter in perimeter_set.label_to_perimeter.items():
                result[label] = perimeter

        return result
