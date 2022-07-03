"""
Directory hierarchies and functions to import data into the bikipy pipeline

Each file defines separate methods
"""
from bikipy.ingress.animal import AnimalIngress
from bikipy.ingress.phase import PhaseIngress

INGRESS_METHOD_NAME_TO_INGRESS_CLASS = {"animal": AnimalIngress, "phase": PhaseIngress}
