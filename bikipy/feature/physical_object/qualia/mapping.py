from bikipy.feature.physical_object.qualia.profile.body_proximity import BodyProximityProfile
from bikipy.feature.physical_object.qualia.profile.field_of_view import FOVCenterToEyesRayCastingProfile

PROFILES = (BodyProximityProfile, FOVCenterToEyesRayCastingProfile)

LABEL_TO_PROFILE = {profile.label: profile for profile in PROFILES}
PROFILE_NAME_TO_PROFILE = {profile.__name__: profile for profile in PROFILES}

PROFILE_MAP = {**LABEL_TO_PROFILE, **PROFILE_NAME_TO_PROFILE}
