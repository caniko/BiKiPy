"""
Directory hierarchies and functions to import data into the bikipy pipeline

Each file defines separate methods
"""
from bikipy.ingress.sequence import sequence_generate_configuration, SequenceIngress

INGRESS_METHOD_NAME_TO_INIT_FUNC = {"sequence": sequence_generate_configuration}
INGRESS_METHOD_NAME_TO_INGRESS_CLASS = {
    "sequence": SequenceIngress
}
