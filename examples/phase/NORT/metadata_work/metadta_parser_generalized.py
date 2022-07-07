import pandas as pd

perimeter_mapper = {"1": 1, "4": 1, "2": 2, "3": 2}

df = pd.read_excel("metadata_raw.ods", index_col=[0, 1])
df.loc[:, "TrialNumber"] = df.loc[:, "Video_file_name"].map(lambda x: x.strip().split(".")[0].split(" ")[1]).astype(int)
df.loc[:, "Perimeter"] = df.loc[:, "Apparatus"].map(lambda x: perimeter_mapper[x.strip().split("-")[-1]])

df.set_index("TrialNumber", append=True, inplace=True)
df.sort_index(inplace=True)

df.to_excel("metadata.xlsx", sheet_name="phase")
