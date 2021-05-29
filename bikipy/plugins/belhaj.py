"""
The following are the functions used to organize data from the Belhaj dataset
"""
import re
from pathlib import Path
from typing import Union

import numpy as np
from pandas import DataFrame

from bikipy.behaviour.nort.experiment import NortExperiment
from bikipy.behaviour.nort.trial import NortObjectField
from bikipy.perimeter.base import PolygonalPerimeter


def _re_pattern_validator(pattern: re.Pattern):
    return pattern if isinstance(pattern, re.Pattern) else re.compile(pattern)


def get_animal_id_vs_exp_ids(
    info_df: DataFrame, exp_id_pattern: Union[re.Pattern, str] = r"\d+"
):
    return (
        info_df[info_df.duplicated("Animal", keep=False)]
        .groupby("Animal")["Test"]
        .apply(tuple)
        .reset_index()
        .set_index("Animal")["Test"]
    )


def get_exp_id_vs_animal_id(id_exp):
    result = {}
    for animal, exps in id_exp.items():
        for exp in exps:
            result[exp] = animal
    return result


def get_animal_id_vs_apparatus(
    info_df, exp_id_pattern: Union[re.Pattern, str] = r"\d+"
):
    exp_id_pattern = _re_pattern_validator(exp_id_pattern)
    animal_id = np.unique(info_df["Animal"])
    apparatus = info_df["Apparatus"]

    return {
        int(i): int(exp_id_pattern.findall(app)[0])
        for i, app in zip(animal_id, apparatus)
    }


def get_exp_id_vs_stage(exp_info_df, exp_id_pattern: Union[re.Pattern, str] = r"\d+"):
    exp_id_pattern = _re_pattern_validator(exp_id_pattern)
    result = {}
    for row in exp_info_df[["Video_file_name", "Stage"]].iterrows():
        exp_idx = int(exp_id_pattern.findall(Path(row[1][0]).stem)[-1])
        stage = Path(row[1][1]).stem.lower()
        result[exp_idx] = stage if stage == "habituation" else stage[-1]

    return result


def round_vs_apparatus_to_general_nort_fields(
    round_vs_field_apparatus: dict, convert_from_legacy: bool = False
):
    rounds = tuple(round_vs_field_apparatus.keys())
    assert len(rounds) == 2  # No novelty -> novelty (two rounds in totalt)

    all_round_fields = [
        tuple(round_vs_field_apparatus[rem_round].keys()) for rem_round in rounds
    ]
    assert all_round_fields.count(all_round_fields[0]) == len(
        all_round_fields
    ), "Rounds have different field designations"
    field_keys = all_round_fields[0]

    result = []
    for field_key in field_keys:
        field_temp_store = {}
        for rem_round in rounds:
            exp_name = NortExperiment.trial_label_to_experiment_class_name[rem_round]
            if exp_name == "training":
                field_temp_store[
                    "constant_object_perimeter"
                ] = round_vs_field_apparatus[rem_round][field_key]["A"]
                field_temp_store[
                    "variable_object_perimeter"
                ] = round_vs_field_apparatus[rem_round][field_key]["B"]
            elif exp_name == "novelty":
                field_temp_store["novel_object_perimeter"] = round_vs_field_apparatus[
                    rem_round
                ][field_key]["B"]

        if convert_from_legacy:
            for key, field in field_temp_store.items():
                kwargs = {"perimeter_corners": field.sides}
                if field.guiding_image:
                    kwargs["inspect_image"] = field.guiding_image
                if field.int_label:
                    kwargs["int_label"] = field.int_label

                field_temp_store[key] = PolygonalPerimeter(**kwargs)

        result.append(NortObjectField(label=field_key, **field_temp_store))

    return result
