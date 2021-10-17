import pandas as pd

a = pd.read_hdf(
    "/mnt/md0/Projects/Neuroscience/Imen/data/y_maze/phd/Y-maze_07.06.2020 (1A)/Test 1DLC_resnet50_y_mazeSep13shuffle1_600000.h5",
    **{
        "index_col": 0,
        "skiprows": 1,
        "header": [0, 1],
        "na_filter": False,
    },
).droplevel(0, axis=1)
print(a)
