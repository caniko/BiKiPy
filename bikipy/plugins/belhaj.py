"""
The following are the functions used to organize data from the Belhaj dataset
"""
import re
from pathlib import Path
from typing import AnyStr, Union

import numpy as np
from pandas import DataFrame


def _re_pattern_validator(pattern: re.Pattern):
    return pattern if isinstance(pattern, re.Pattern) else re.compile(pattern)


def get_animal_id_vs_exp_ids(info_df: DataFrame, exp_id_pattern: Union[re.Pattern, AnyStr] = r"\d+"):
    exp_id_pattern = _re_pattern_validator(exp_id_pattern)
    exp_ids = np.array([int(exp_id_pattern.findall(info)[-1]) for info in info_df["Video_file_name"]])
    animal_id = np.array(info_df["Animal"])

    result = {}
    for i in range(int(animal_id.min()), int(animal_id.max() + 1)):
        loc = np.where(animal_id == i)[0][:2]
        result[i] = tuple(exp_ids[loc])

    return result


def get_exp_id_vs_animal_id(id_exp):
    result = {}
    for animal, exps in id_exp.items():
        for exp in exps:
            result[exp] = animal
    return result


def get_animal_id_vs_apparatus(info_df, exp_id_pattern: Union[re.Pattern, AnyStr] = r"\d+"):
    exp_id_pattern = _re_pattern_validator(exp_id_pattern)
    animal_id = np.unique(info_df["Animal"])
    apparatus = info_df["Apparatus"]

    return {
        int(i): int(exp_id_pattern.findall(app)[0])
        for i, app
        in zip(animal_id, apparatus)
    }


def get_exp_id_vs_stage(exp_info_df, exp_id_pattern: Union[re.Pattern, AnyStr] = r"\d+"):
    exp_id_pattern = _re_pattern_validator(exp_id_pattern)
    result = {}
    for row in exp_info_df[['Video_file_name', 'Stage']].iterrows():
        exp_idx = int(exp_id_pattern.findall(Path(row[1][0]).stem)[-1])
        stage = Path(row[1][1]).stem.lower()
        result[exp_idx] = stage if stage == "habituation" else stage[-1]

    return result
