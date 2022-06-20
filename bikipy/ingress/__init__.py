"""
Directory hierarchies and functions to import data into the bikipy pipeline

Each file defines separate methods
"""
from bikipy.ingress.animal import AnimalIngress, sequence_generate_configuration

INGRESS_METHOD_NAME_TO_INIT_FUNC = {"animal": sequence_generate_configuration}
INGRESS_METHOD_NAME_TO_INGRESS_CLASS = {"animal": AnimalIngress}
