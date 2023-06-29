from bikipy.feature.qualia.physical_object.heuristic.abc import HeuristicCLS
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

SOLO_HEURISTICS = (
    BodyProximityHeuristic,
    WhiskerInteractionHeuristic,
    OlfactionHeuristic,
)
HELPER_HEURISTICS = (OutsideObjectPerimeterHeuristic,)

HEURISTICS = (*SOLO_HEURISTICS, *HELPER_HEURISTICS)
HEURISTIC_NAME_TO_HEURISTIC = {heuristic.__name__: heuristic for heuristic in HEURISTICS}


def map_alias_to_heuristic(heuristics: tuple[HeuristicCLS, ...]) -> dict[str, HeuristicCLS]:
    return {heuristic.heuristic_alias: heuristic for heuristic in heuristics}


alias_to_solo_heuristic = map_alias_to_heuristic(SOLO_HEURISTICS)
alias_to_helper_heuristic = map_alias_to_heuristic(HELPER_HEURISTICS)

label_to_heuristics = {**alias_to_solo_heuristic, **alias_to_helper_heuristic}
alias_to_heuristics_cls = {**label_to_heuristics, **HEURISTIC_NAME_TO_HEURISTIC}
