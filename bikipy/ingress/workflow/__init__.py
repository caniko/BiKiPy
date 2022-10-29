"""
Directory hierarchies and functions to import data into the bikipy pipeline

Each file defines separate methods
"""
from bikipy.ingress.workflow.animal import AnimalIngress
from bikipy.ingress.workflow.animal_day import AnimalDayIngress
from bikipy.ingress.workflow.phase import PhaseIngress

INGRESS_METHOD_NAME_TO_INGRESS_CLASS = {"animal": AnimalIngress, "animal_day": AnimalDayIngress, "phase": PhaseIngress}
