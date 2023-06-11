import numpy as np
from formulaic.parser import DefaultFormulaParser
from formulaic.parser.types import Token


dsl = Grammar()


def parse_heuristic_formula(
    formula: str, name2boolean_index: dict[str, np.ndarray[bool, bool]]
) -> np.ndarray[bool, bool]:
    """
    Parse the formula, and return the combined boolean index.

    :param formula:
    :param name2boolean_index:
    :return:
    """
    and_sign = False
    or_sign = False
    flip_sign = False

    current_boolean_index = None

    for token in DefaultFormulaParser(include_intercept=False).get_tokens(formula):
        match token.kind.value:
            case "name":
                try:
                    upcoming_merge = name2boolean_index[token.token]
                except KeyError as e:
                    msg = f"{token.token} is not defined; pick from: {', '.join(name2boolean_index)}"
                    raise KeyError(msg) from e

                if flip_sign:
                    upcoming_merge = ~upcoming_merge
                    flip_sign = False

                if current_boolean_index is not None:
                    if and_sign:
                        assert not or_sign
                        current_boolean_index = current_boolean_index & upcoming_merge
                        and_sign = False
                    elif or_sign:
                        current_boolean_index = current_boolean_index | upcoming_merge
                        or_sign = False
                    else:
                        msg = f"No signs detected before merging two boolean indices: {formula}"
                        raise ValueError(msg)
                else:
                    current_boolean_index = upcoming_merge

            case "operator":
                match token.token:
                    case "~":
                        flip_sign = True
                    case "&":
                        and_sign = True
                    case "|":
                        or_sign = True
                    case _:
                        msg = f"Unsupported operator {token.token}"
                        raise ValueError(msg)

            case _:
                msg = f"Unsupported token type {token}"
                raise ValueError(msg)

    assert not (and_sign or or_sign or flip_sign), f"Dangling operators: {formula}"
    assert current_boolean_index is not None, f"No boolean indices defined: {formula}"

    return current_boolean_index
