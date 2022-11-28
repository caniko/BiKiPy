import numpy as np
import pandas as pd

perimeter_mapper = {"1": 1, "4": 1, "2": 2, "3": 2}

df = pd.read_excel("metadata_raw.ods", index_col=[0, 1])
df.loc[:, "TrialNumber"] = df.loc[:, "Video_file_name"].map(lambda x: x.strip().split(".")[0].split(" ")[1]).astype(int)

# Perimeter assignment
df.loc[:, "Apparatus"] = df.loc[:, "Apparatus"].map(lambda x: x.strip().split("-")[2]).astype(int)


def a1_reference_mapper(ser):
    test = ser["Stage"]
    if test == 0:
        return np.nan
    result = f"{test_mapper[test]}_{ser['Apparatus']}"
    if result == "training_1":
        return np.nan
    return result


def a2_reference_mapper(ser):
    test = ser["Stage"]
    if test == 0:
        return np.nan
    result = f"{test_mapper[test]}_{ser['Apparatus']}"
    if result != "training_1":
        return np.nan
    return result


test_mapper = {1: "training", 2: "novel"}
df.loc[pd.IndexSlice["A", 1], "Perimeter"] = df.loc[pd.IndexSlice["A", 1], ["Apparatus", "Stage"]].apply(
    a1_reference_mapper, axis=1
)
df.loc[pd.IndexSlice["A", 1], "ChangeReference"] = df.loc[pd.IndexSlice["A", 1], ["Apparatus", "Stage"]].apply(
    a2_reference_mapper, axis=1
)


df.loc[pd.IndexSlice["A", 2], "Perimeter"] = df.loc[pd.IndexSlice["A", 2], ["Apparatus", "Stage"]].apply(
    a2_reference_mapper, axis=1
)


def reference_mapper(ser):
    test = ser["Stage"]
    if test == 0:
        return np.nan
    return f"{test_mapper[test]}_{ser['Apparatus']}"


df.loc[pd.IndexSlice["A", 2], "ChangeReference"] = df.loc[pd.IndexSlice["A", 2], ["Apparatus", "Stage"]].apply(
    a1_reference_mapper, axis=1
)
df.loc[pd.IndexSlice["B", :], "ChangeReference"] = df.loc[pd.IndexSlice["B", :], ["Apparatus", "Stage"]].apply(
    reference_mapper, axis=1
)


def image_name_mapper(ser):
    test = ser["Stage"]
    if test == 0:
        return np.nan
    return f"{phase_part}_{test_mapper[test]}_{ser['Apparatus']}"


phase_part = "A1"
df.loc[pd.IndexSlice["A", 1], "ChangeReferenceImageName"] = df.loc[pd.IndexSlice["A", 1], ["Apparatus", "Stage"]].apply(
    image_name_mapper, axis=1
)
phase_part = "A2"
df.loc[pd.IndexSlice["A", 2], "ChangeReferenceImageName"] = df.loc[pd.IndexSlice["A", 2], ["Apparatus", "Stage"]].apply(
    image_name_mapper, axis=1
)
phase_part = "B1"
df.loc[pd.IndexSlice["B", 1], "ChangeReferenceImageName"] = df.loc[pd.IndexSlice["B", 1], ["Apparatus", "Stage"]].apply(
    image_name_mapper, axis=1
)
phase_part = "B2"
df.loc[pd.IndexSlice["B", 2], "ChangeReferenceImageName"] = df.loc[pd.IndexSlice["B", 2], ["Apparatus", "Stage"]].apply(
    image_name_mapper, axis=1
)

df.set_index("TrialNumber", append=True, inplace=True)
df.sort_index(inplace=True)
df.to_excel("metadata.xlsx", sheet_name="phase")
