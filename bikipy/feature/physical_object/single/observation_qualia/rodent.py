from bikipy.feature.physical_object.single.component.gaze import GazeComponent
from bikipy.feature.physical_object.single.component.nose_tail_proximity import NoseTailProximity
from bikipy.feature.physical_object.single.component.olfaction import OlfactionComponent
from bikipy.feature.physical_object.single.observation_qualia.abc import AbcPhysicalObjectObservationQualia


class RodentObservationQualia(AbcPhysicalObjectObservationQualia[GazeComponent, OlfactionComponent]):
    observation_component_classes = (GazeComponent, OlfactionComponent)


class RodentFullBodyObservationQualia(AbcPhysicalObjectObservationQualia[GazeComponent, NoseTailProximity]):
    observation_component_classes = (GazeComponent, NoseTailProximity)
