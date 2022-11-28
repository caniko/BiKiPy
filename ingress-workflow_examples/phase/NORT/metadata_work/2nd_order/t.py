import shutil
from pathlib import Path

import numpy as np
import pandas as pd

COLUMN_NAMES_TO_REMOVE = ("Perimeter", "ChangeReference", "ChangeReferenceImageName")


metadata_df = pd.read_excel("metadata.xlsx", index_col=[0, 1, 2])
output = Path("output")

for f in Path("trialwise").glob("**/*.csv"):
    phase_str, phase_idx = f.parent.stem
    trial_id = int(f.stem.split("-")[0])
    new_name = f"{phase_str}{phase_idx}_{trial_id}"

    metadata_df.loc[(phase_str, int(phase_idx), trial_id), COLUMN_NAMES_TO_REMOVE] = (new_name, np.nan, np.nan)

    shutil.copy(f, output / f"perimeter-rectangle-{new_name}.csv")


metadata_df.to_excel("metadata_new.xlsx")
