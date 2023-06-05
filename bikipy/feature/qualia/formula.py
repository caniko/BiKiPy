# import numpy as np
# from formulaic.parser import DefaultFormulaParser
# from formulaic.parser.types import Token
#
#
# def _unsupported_exception_raise(token: Token):
#     msg = f"Unsupported operator {token.token}"
#     raise ValueError(msg)
#
#
# def parse_heuristic_formula(formula: str, name2boolean_index: dict[str, np.ndarray[bool, bool]]):
#     """
#     Parse the formula, and return the combined boolean index.
#
#     :param formula:
#     :param name2boolean_index:
#     :return:
#     """
#     result = None
#     for name, boolean_index in name2boolean_index.items():
#         if result is None:
#             result = boolean_index
#         else:
#             assert result.shape == boolean_index.shape, "Boolean indices must have the same shape"
#
#     and_sign = False
#     or_sign = False
#     flip_sign = False
#     current_boolean_index = None
#     for token in DefaultFormulaParser(include_intercept=False).get_tokens(formula):
#         match token.kind.value:
#             case "name":
#                 if current_operator
#                 else:
#                     current_boolean_index = name2boolean_index[token.token]
#
#             case "operator":
#                 match token.token:
#                     case "~":
#                         current_boolean_index = ~current_boolean_index
#                     case _:
#                         _unsupported_exception_raise(token)
#                 current_operator = token.token
#
#             case _:
#                 _unsupported_exception_raise(token)
