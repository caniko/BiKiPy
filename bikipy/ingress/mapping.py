from bikipy.ingress.sequence import (
    sequence_analysis_keyword_arguments,
    sequence_generate_configuration,
)

INGRESS_METHOD_NAME_TO_INIT_FUNC = {"sequence": sequence_generate_configuration}
INGRESS_METHOD_NAME_TO_KEYWORD_ARGUMENT_FUNC = {"sequence": sequence_analysis_keyword_arguments}
