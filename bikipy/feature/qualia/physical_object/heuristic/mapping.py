from bikipy.feature.qualia.physical_object.heuristic.body_proximity import (
    BodyProximityHeuristic,
)
from bikipy.feature.qualia.physical_object.heuristic.field_of_view import (
    FOVCenterToEyesRayCastingHeuristic,
)
from bikipy.feature.qualia.physical_object.heuristic.olfaction import OlfactionHeuristic

HEURISTICS = (BodyProximityHeuristic, FOVCenterToEyesRayCastingHeuristic, OlfactionHeuristic)

LABEL_TO_HEURISTIC = {heuristic.heuristic_alias: heuristic for heuristic in HEURISTICS}
HEURISTIC_NAME_TO_HEURISTIC = {heuristic.__name__: heuristic for heuristic in HEURISTICS}

HEURISTIC_MAP = {**LABEL_TO_HEURISTIC, **HEURISTIC_NAME_TO_HEURISTIC}
