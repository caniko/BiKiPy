from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np
from pydantic import DirectoryPath, validate_arguments
from pydantic_numpy import NDArrayBool

from bikipy.feature.compute import ComputeBooleanIndex
from bikipy.core.video import VideoMetadataMixin
from bikipy.feature.physical_object.qualia.profile.abc import QualiaProfile
from bikipy.feature.tolerance.plural import plural_node_tolerance_model

# TODO: WIP


class JITQualiaProfile(VideoMetadataMixin):
    qualia_heuristic_sequence: list[QualiaProfile | ComputeBooleanIndex, ...]
    tolerance_model: bool = True

    @cached_property
    def qualia_index(self) -> NDArrayBool:
        if len(self.qualia_heuristic_sequence) == 1:
            return self.qualia_heuristic_sequence[0].result
        return (
            plural_node_tolerance_model(*self.qualia_heuristic_sequence, fps=self.video.fps)
            if self.tolerance_model
            else np.logical_and.reduce((feat.result for feat in self.qualia_heuristic_sequence))
        )

    @validate_arguments
    def plot(self, save_directory: DirectoryPath) -> None:
        if len(self.qualia_heuristic_sequence) == 1:
            self.qualia_heuristic_sequence[0].plot()
            plt.savefig(save_directory / "qualia_heuristic.jpeg")
            plt.close()
        else:
            for idx, heuristic in enumerate(self.qualia_heuristic_sequence):
                heuristic.plot()
                plt.savefig(save_directory / f"{idx}_qualia_heuristic.jpeg")
                plt.close()
