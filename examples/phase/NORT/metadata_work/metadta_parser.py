import numpy as np
import pandas as pd

perimeter_mapper = {"1": 1, "4": 1, "2": 2, "3": 2}

df = pd.read_excel("metadata_raw.ods", index_col=[0, 1])
df.loc[:, "TrialNumber"] = df.loc[:, "Video_file_name"].map(lambda x: x.strip().split(".")[0].split(" ")[1]).astype(int)

# Perimeter assignment
df.loc[:, "Apparatus"] = df.loc[:, "Apparatus"].map(lambda x: x.strip().split("-")[2]).astype(int)


def a1_mapper(ser):
    test = ser["Test"]
    if test == 0:
        return np.nan
    return f"{test_mapper[test]}_{ser['Apparatus']}"


test_mapper = {1: "training", 2: "novel"}
df.loc[pd.IndexSlice["A", 1], "Perimeter"] = df.loc[pd.IndexSlice["A", 1], ["Apparatus", "Test"]].apply(
    a1_mapper, axis=1
)

df.set_index("TrialNumber", append=True, inplace=True)
df.sort_index(inplace=True)
df.to_excel("metadata.xlsx", sheet_name="phase")
