"""
This is for re-referencing perimeters
"""
from typing import Any

from bikipy.core.typing import NDArrayFp64
from bikipy.ingress.plugin.base import BasePluginFile
from bikipy.utils.io.makesense import image_name_to_point_from_makesense


class PluginReference(BasePluginFile):
    ingress_key = "reference_definition_strategy"
    code_key = "reference"
    human_readable_index = "Reference"

    @property
    def bikipy_trial_key(self) -> str:
        ...

    def image_name_to_re_referencing_point(self) -> dict[str, NDArrayFp64]:
        if (path_to_re_reference_file := self.data_path.parent / f"re_reference-{self.label}.csv").exists():
            assert self.reference_point is not None, "Reference point must be defined for re-referencing"
            return image_name_to_point_from_makesense(path_to_re_reference_file)
        return {}
