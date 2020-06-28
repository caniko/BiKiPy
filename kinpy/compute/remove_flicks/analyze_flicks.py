from itertools import combinations
import numpy as np


from ._event_data_getters import *


def get_velocity(df, body_part):
    x = df.loc[:, (body_part, "x")].interpolate(method="linear").values
    y = df.loc[:, (body_part, "y")].interpolate(method="linear").values

    return np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)


def get_position(df, body_part):
    x = df.loc[:, (body_part, "x")].interpolate(method="linear").values
    y = df.loc[:, (body_part, "y")].interpolate(method="linear").values

    return np.sqrt(x ** 2 + y ** 2)


def invalid_relative_part_distance(df, md_tol=1.5):
    max_index = df.index[-1]
    new_df = df.copy()
    for pair in combinations(df.columns.levels[0], 2):
        rpd_values = np.zeros(max_index, dtype=bool)
        pos_a = get_position(df, pair[0])
        pos_b = get_position(df, pair[1])

        if pos_a.size != pos_b.size:
            msg = "Array a and b have different lengths," "{} and {}".format(
                pos_a.length, pos_b.length
            )
            raise ValueError(msg)

        dist_ab = np.abs(pos_a - pos_b)
        median_length = np.median(dist_ab)
        irpd_locs = np.where(dist_ab > median_length * md_tol)[0]

        new_df.loc[irpd_locs, [(pair[0], "x"), (pair[0], "y")]] = np.nan
        new_df.loc[irpd_locs, [(pair[1], "x"), (pair[1], "y")]] = np.nan

    return new_df


def find_high_velocity_events(df, body_part, max_vel, fps=30):
    position = get_position(df, body_part)
    velocity = get_velocity(df, body_part)

    pos_len = len(position)
    hv_locs = np.zeros(pos_len + 1, dtype=bool)

    hv_index = np.where(velocity > max_vel)[0]

    if hv_index.size == 1:
        event = hv_index[0]
        hv_locs[event + 1] = check_lone_hv(event, velocity, max_vel, fps)
        return hv_locs

    hv_index_interval = np.diff(hv_index)

    index = 0
    while index < hv_index_interval.size:
        event_start_index = index
        event_end_index = get_event_end_index(event_start_index, hv_index, fps)
        before_ev = get_before_event(
            event_start_index, event_end_index, position, hv_index, fps
        )
        following_ev = get_following_event(
            event_start_index,
            event_end_index,
            position,
            hv_index_interval.size,
            hv_index,
            fps,
        )

        if abs(before_ev - following_ev) < max_vel:
            hv_locs[event_start_index:event_end_index] = True
        index += 1


def get_flicks(df, body_part, max_vel):
    velocity = get_velocity(df, body_part)
    flick_locs = np.zeros(len(velocity) + 1, dtype=bool)

    hv_locs = np.where(velocity > max_vel)[0]
    if not hv_locs.size:
        return flick_locs

    flick_loc_intervals = np.diff(hv_locs)

    flick_elements = hv_locs[np.where(flick_loc_intervals == 1)]
    flick_elements = np.sort(
        np.concatenate((flick_elements, flick_elements + 1), axis=None)
    )
    flick_locs[flick_elements + 1] = True

    return flick_locs
