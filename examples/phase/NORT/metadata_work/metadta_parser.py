import pandas as pd

perimeter_mapper = {"1": 1, "4": 1, "2": 2, "3": 2}

df = pd.read_excel("metadata_raw.ods", index_col=[0, 1, 2]).sort_index()
df.loc[:, "TrialNumber"] = df.loc[:, "Video_file_name"].map(lambda x: x.strip().split(".")[0].split(" ")[1])
df.loc[:, "Perimeter"] = df.loc[:, "Apparatus"].map(lambda x: perimeter_mapper[x.strip().split("-")[-1]])
df.to_excel("metadata.xlsx")
