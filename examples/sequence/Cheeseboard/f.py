import pandas as pd

def mani(x: str):
    return x.replace("D", "")

df = pd.read_excel("metadata.xlsx")
df.loc[:, "directory"] = df.loc[:, "directory"].apply(mani)K
1
