import logging
import os
from functools import lru_cache
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd
from pydantic import DirectoryPath, validate_call

from bikipy.utils.constants import TO_PARQUET_KWARGS

logger = logging.getLogger(__name__)


@lru_cache
def compute_midpoint_label(midpoint_group: Iterable[str], manual_midpoint_label: Optional[str] = None) -> str:
    return manual_midpoint_label or f"{'_'.join(midpoint_group)}_midpoint"


def read_bonsai_timestamps(file_path):
    datetime_array = (
        pd.read_csv(file_path, header=None, usecols=[16], parse_dates=[0]).values.T[0].astype(np.datetime64)
    )
    return (datetime_array - datetime_array[0]).astype(float) / 10**6


@validate_call
def merge_timestamps_with_dlc(
    dataset_dir: DirectoryPath,
    timestamp_reader: Callable = read_bonsai_timestamps,
    timestamp_file_lookup_expression: str = "*timestamps-*.csv",
    coordinate_file_lookup_expression: str = "*.parquet",
    delimiter: str = ".",
) -> None:
    from bikipy.reader.data_with_likelihood import DataWithLikelihoodReader

    def get_first_delimited_value_from_str(string: str):
        return string.split(delimiter)[0]

    for dataset_unit_directory_name in os.listdir(dataset_dir):
        print(dataset_unit_directory_name)
        dataset_unit_dir = dataset_dir / str(dataset_unit_directory_name)

        label_to_timestamp, label_to_timestamp_path = {}, {}
        for timestamp_file in dataset_unit_dir.glob(timestamp_file_lookup_expression):
            label = get_first_delimited_value_from_str(timestamp_file.stem)
            label_to_timestamp[label] = timestamp_reader(timestamp_file)
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
            df.to_parquet(timestamped_df_path, **TO_PARQUET_KWARGS)

            os.remove(label_to_timestamp_path[timestamp_label])
            os.remove(coord_file)


def trial_video_frame_slice(
    likelihoods: pd.DataFrame,
    required_tail_likelihood: float,
    crop_target_trial_length_frames: int,
    frames_to_try_to_crop_from_start: Optional[int] = None,
    frames_to_try_to_crop_from_end: Optional[int] = None,
    crop_target_from_end: bool = True,
) -> slice:
    combined_raw_likelihood = likelihoods.mean(axis=1).values

    valid_likelihood_index = np.where(combined_raw_likelihood >= required_tail_likelihood)[0]

    # first_full_body_detection_frame_index
    start_frame = int(valid_likelihood_index[0])
    # last_full_body_detection_frame_index
    last_frame = int(valid_likelihood_index[-1])

    raw_duration_frames = last_frame - start_frame

    naive_start = start_frame
    naive_end = last_frame
    frames_to_try_to_crop = 0
    if frames_to_try_to_crop_from_start:
        frames_to_try_to_crop += frames_to_try_to_crop_from_start
        naive_start += frames_to_try_to_crop_from_start

    if frames_to_try_to_crop_from_end:
        frames_to_try_to_crop += frames_to_try_to_crop_from_end
        naive_end -= frames_to_try_to_crop_from_end

    crop_minus_duration = raw_duration_frames - frames_to_try_to_crop

    if crop_minus_duration == crop_target_trial_length_frames:
        return slice(naive_start, naive_end)

    if crop_minus_duration < crop_target_trial_length_frames:
        logger.warning(
            (
                f"Trial length ({crop_target_trial_length_frames}) is greater than the "
                f"duration_frames of the video ({raw_duration_frames - frames_to_try_to_crop})."
            )
        )

    rest_to_target = crop_minus_duration - crop_target_trial_length_frames

    if crop_target_from_end:
        crop_start_frames = naive_start
        crop_end_frames = naive_end - rest_to_target

    else:
        crop_start_frames = naive_start + rest_to_target
        crop_end_frames = naive_end

    return slice(crop_start_frames, crop_end_frames)
