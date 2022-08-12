import os
from functools import lru_cache
from typing import Callable, Iterable, Optional

from pydantic import DirectoryPath, validate_arguments


@lru_cache
def compute_midpoint_label(midpoint_group: Iterable[str], manual_midpoint_label: Optional[str] = None) -> str:
    return manual_midpoint_label or f"{'_'.join(midpoint_group)}_midpoint"


@validate_arguments
def merge_timestamps_with_dlc(
    dataset_dir: DirectoryPath,
    file_to_timestamp_series: Callable,
    timestamp_file_lookup_expression: str = "*timestamps-*.csv",
    coordinate_file_lookup_expression: str = "*.parquet",
    delimiter: str = ".",
):
    from bikipy.reader.data_with_likelihood import DataWithLikelihoodReader

    def get_first_delimited_value_from_str(string: str):
        return string.split(delimiter)[0]

    for dataset_unit_directory_name in os.listdir(dataset_dir):
        print(dataset_unit_directory_name)
        dataset_unit_dir = dataset_dir / str(dataset_unit_directory_name)

        label_to_timestamp, label_to_timestamp_path = {}, {}
        for timestamp_file in dataset_unit_dir.glob(timestamp_file_lookup_expression):
            label = get_first_delimited_value_from_str(timestamp_file.stem)
            label_to_timestamp[label] = file_to_timestamp_series(timestamp_file)
            label_to_timestamp_path[label] = timestamp_file

        for coord_file in dataset_unit_dir.glob(coordinate_file_lookup_expression):
            timestamped_df_path = coord_file.with_name(f"{coord_file.stem}-timestamped.parquet")
            if (
                timestamped_df_path.exists()
                or get_first_delimited_value_from_str(coord_file.stem) not in label_to_timestamp
            ):
                continue

            timestamp_label = get_first_delimited_value_from_str(coord_file.stem)

            df = DataWithLikelihoodReader(df_path=coord_file).raw_df
            if len(label_to_timestamp[timestamp_label]) != len(df):
                os.remove(label_to_timestamp_path[timestamp_label])
                continue

            df.set_index(label_to_timestamp[timestamp_label], inplace=True)
            df.to_parquet(timestamped_df_path)

            os.remove(label_to_timestamp_path[timestamp_label])
            os.remove(coord_file)
