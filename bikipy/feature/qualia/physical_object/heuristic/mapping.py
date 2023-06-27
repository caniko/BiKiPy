from bikipy.feature.qualia.physical_object.heuristic.body_proximity import (
    BodyProximityHeuristic,
)
from bikipy.feature.qualia.physical_object.heuristic.olfaction import OlfactionHeuristic
from bikipy.feature.qualia.physical_object.heuristic.whiskers import (
    WhiskerInteractionHeuristic,
)

HEURISTICS = (BodyProximityHeuristic, WhiskerInteractionHeuristic, OlfactionHeuristic)

LABEL_TO_HEURISTIC = {heuristic.heuristic_alias: heuristic for heuristic in HEURISTICS}
HEURISTIC_NAME_TO_HEURISTIC = {heuristic.__name__: heuristic for heuristic in HEURISTICS}

ALIAS_TO_HEURISTIC_CLS = {**LABEL_TO_HEURISTIC, **HEURISTIC_NAME_TO_HEURISTIC}
