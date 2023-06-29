from bikipy.feature.qualia.physical_object.heuristic.helper.outside_object_perimeter import (
    OutsideObjectPerimeterHeuristic,
)
from bikipy.feature.qualia.physical_object.heuristic.solo.body_proximity import (
    BodyProximityHeuristic,
)
from bikipy.feature.qualia.physical_object.heuristic.solo.olfaction import (
    OlfactionHeuristic,
)
from bikipy.feature.qualia.physical_object.heuristic.solo.whiskers import (
    WhiskerInteractionHeuristic,
)

HEURISTICS = (
    # Solo
    BodyProximityHeuristic,
    WhiskerInteractionHeuristic,
    OlfactionHeuristic,
    # Helper
    OutsideObjectPerimeterHeuristic,
)

LABEL_TO_HEURISTIC = {heuristic.heuristic_alias: heuristic for heuristic in HEURISTICS}
HEURISTIC_NAME_TO_HEURISTIC = {heuristic.__name__: heuristic for heuristic in HEURISTICS}

ALIAS_TO_HEURISTIC_CLS = {**LABEL_TO_HEURISTIC, **HEURISTIC_NAME_TO_HEURISTIC}
