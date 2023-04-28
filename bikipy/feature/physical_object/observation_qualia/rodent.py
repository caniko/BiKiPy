from bikipy.feature.physical_object.component.gaze import CenterToEyesRayCasting
from bikipy.feature.physical_object.component.olfaction import OlfactionComponent
from bikipy.feature.physical_object.observation_qualia.abc import AbcPhysicalObjectObservationQualia


class RodentObservationQualia(AbcPhysicalObjectObservationQualia[CenterToEyesRayCasting, OlfactionComponent]):
    observation_component_classes =
