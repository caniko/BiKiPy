from glob import glob
from pathlib import Path

from bikipy.reader.deeplabcut import convert_hdf_to_parquet


h5_files = Path("dataset/").resolve().glob("**/*.h5")
for f in h5_files:
    convert_hdf_to_parquet(f, True)
