from functools import cached_property
from typing import Sequence

import numpy as np

from bikipy.feature.compute import ComputeBooleanIndex
from bikipy.core.video import VideoMetadataMixin
from bikipy.feature.physical_object.qualia.component.abc import QualiaComponent
from bikipy.feature.tolerance.plural import plural_node_tolerance_model


class QualiaProfile(VideoMetadataMixin):
    qualia_component_sequence: Sequence[QualiaComponent | ComputeBooleanIndex]
    tolerance_model: bool = True

    @cached_property
    def qualia_index(self):
        return (
            plural_node_tolerance_model(*self.qualia_component_sequence, fps=self.video.fps)
            if self.tolerance_model
            else np.logical_and.reduce((feat.result for feat in self.qualia_component_sequence))
        )
