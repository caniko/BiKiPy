from bikipy.feature.qualia.physical_object.qualia_heuristic.body_proximity import (
    BodyProximityProfile,
)
from bikipy.feature.qualia.physical_object.qualia_heuristic.field_of_view import (
    FOVCenterToEyesRayCastingProfile,
)

PROFILES = (BodyProximityProfile, FOVCenterToEyesRayCastingProfile)

LABEL_TO_PROFILE = {heuristic.label: heuristic for heuristic in PROFILES}
PROFILE_NAME_TO_PROFILE = {heuristic.__name__: heuristic for heuristic in PROFILES}

PROFILE_MAP = {**LABEL_TO_PROFILE, **PROFILE_NAME_TO_PROFILE}
