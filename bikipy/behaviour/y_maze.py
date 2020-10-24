from typing import Union, AnyStr, Sequence, List

import numpy as np


def reduce_location_sequence(location_per_frame: Sequence) -> List:
    """
    Reduce the location per frame to a location sequence

    Parameters
    ----------
    location_per_frame
        Sequence of location on the respective frame

    Returns
    -------
    List containing the location sequence; (A, A, A, B, B, C) -> [A, B, C]
    """

    current_char = None
    arm_location_sequence = []
    for location in location_per_frame:
        if isinstance(location, str) and (
            location != current_char or not current_char
        ):
            arm_location_sequence.append((current_char := location))

    return arm_location_sequence


def total_alternations(arm_location_sequence: Sequence) -> int:
    return len(arm_location_sequence) - 2


def spontaneous_alterntations(
    arm_location_sequence: Sequence, exclude: Union[AnyStr, None] = None
) -> float:
    """
    Define the number of spontaneous alternations between each y-maze arm

    Parameters
    ----------
    arm_location_sequence
        Sequence of arm locations; A, B, A, C, etc

    Returns
    -------
    float, defining the percentage ratio between alternations and possible number
    of maximum alternations
    """

    if exclude:
        arm_location_sequence = [x for x in arm_location_sequence if x != exclude]

    alternations = 0
    total_alternations_exp = total_alternations(arm_location_sequence)
    for i in range(total_alternations_exp):
        current_string = f"{arm_location_sequence[i]}{arm_location_sequence[i+1]}{arm_location_sequence[i+2]}"
        if "A" in current_string and "B" in current_string and "C" in current_string:
            alternations += 1

    return 100.0 * alternations / total_alternations_exp
