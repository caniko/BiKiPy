from bikipy.core.typing import TrialId
from bikipy.core.video import VideoMetadata
from bikipy.feature.physical_object.single.component.gaze import GazeComponent
from bikipy.feature.physical_object.single.component.olfaction import OlfactionComponent
from bikipy.feature.physical_object.single.profile.abc import AbcPhysicalObjectProfile


class RodentProfile(AbcPhysicalObjectProfile):
    @classmethod
    def with_components(
        cls, label: str, trial_obj_label: TrialId, video: VideoMetadata, **component_fields
    ) -> "RodentProfile":
        return cls(
            observation_components=(
                GazeComponent(manual_video=video, **component_fields),
                OlfactionComponent(manual_video=video, **component_fields),
            ),
            label=label,
            trial_obj_label=trial_obj_label,
            manual_video=video,
        )
