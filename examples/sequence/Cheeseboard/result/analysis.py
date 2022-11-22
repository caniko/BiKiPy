from collections import defaultdict
from functools import partial

import numpy as np
import pandas as pd


for directory in ("circle-5cm/no_tol", "circle-5cm/tol", "circle-10cm/no_tol", "circle-10cm/tol"):
    df = pd.read_parquet(f"{directory}/cheeseboard_reward_trace-cheeseboard.parquet")

    dict_dict_list = partial(defaultdict, list)

    reward_trace_raw, in_reward_area_raw, distance_raw, probe_records = (
        defaultdict(dict_dict_list),
        defaultdict(dict_dict_list),
        defaultdict(dict_dict_list),
        {},
    )
    for trial_id, row in df.iterrows():
        animal, day, daily_id = trial_id.split("_")

        animal = int(animal)
        day_day_id = f"{day}_{daily_id}"
        if day_day_id == "6_1" or day_day_id == "13_1":
            probe_records[(animal, day_day_id)] = row
        else:
            day = int(day)
            reward_trace_raw[animal][day].append(row[("Cheeseboard", "RewardTraceSeconds")])
            in_reward_area_raw[animal][day].append(row[("Cheeseboard", "RewardAreaSeconds")])
            distance_raw[animal][day].append(row[("StartToReward", "Displacement")])

    def average_day(raw_data):
        result = defaultdict(dict)
        for animal, days in raw_data.items():
            for day, data in days.items():
                result[(animal, day)] = np.nanmean(data)
        return result

    reward_trace_records = pd.DataFrame.from_dict(
        average_day(reward_trace_raw), orient="index", columns=["RewardTrace"]
    )
    in_reward_area_records = pd.DataFrame.from_dict(
        average_day(in_reward_area_raw), orient="index", columns=["InRewardArea"]
    )
    distance_records = pd.DataFrame.from_dict(average_day(distance_raw), orient="index", columns=["DistanceToReward"])

    df = pd.concat([reward_trace_records, in_reward_area_records, distance_records], axis=1, join="inner")
    df.index = pd.MultiIndex.from_tuples(df.index, names=("Animal", "Day"))

    df.to_excel(f"{directory}/result.xlsx")
