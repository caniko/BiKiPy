from abc import ABC
from typing import Optional

from pydantic import PositiveInt

from bikipy.ingress.plugin.base import BasePluginFile, HasReferenceMixin
from bikipy.perimeter.base import (
    SinglePerimeter,
    StringPerimeterShapes,
    perimeter_set_from_makesense,
)


class AbcPerimeterPlugin(BasePluginFile, HasReferenceMixin, ABC):
    manual_shape: Optional[StringPerimeterShapes] = None
    warn_missing_re_reference_file: bool = False

    plural_entries = True

    ingress_key = "perimeter"
    code_key = "perimeter"
    default_trial_argument_key = "label_to_perimeter"
    human_readable_index = "Perimeter"

    def perimeter_mapper(self, trial_id: Optional[str | PositiveInt] = None) -> dict[str, SinglePerimeter]:
        additional_kwargs = {**self.perimeter_settings["common"]["defined"]}
        if self.perimeter_settings["trial_perimeters"]:
            additional_kwargs.update(
                self._parse_plugin_settings(
                    self.perimeter_settings["trial_perimeters"][self.trial_argument_key]["defined"]
                )
            )

        image_name_to_perimeter_set = perimeter_set_from_makesense(
            self.data_path,
            self.manual_shape or self.shape,
            init_args=self.init_args,
            meters_per_pixel=self.ingress.get_meter_per_pixel(trial_id),
            reference_point_array=self.reference_point,
            inspect_arg=self.ingress.inspect_directory_path,
            **additional_kwargs,
        )

        result = {}
        for image_name, perimeter_set in image_name_to_perimeter_set.items():
            for label, perimeter in perimeter_set.label_to_perimeter.items():
                result[label] = perimeter

        return result
