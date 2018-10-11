import numpy as np
import pandas as pd

nan = np.nan
df = pd.DataFrame.from_dict({'col1': (10, nan, nan, 5)})
df_int = df.interpolate(method='linear', limit_area='inside')

print(df_int)
