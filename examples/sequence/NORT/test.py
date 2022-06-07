import pandas as pd


df = pd.read_excel("metadata.xlsx")
df.set_index("Animal", inplace=True)

row = df.loc["OO670", :]

1
