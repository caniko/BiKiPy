import numpy as np


def check_lone_hv(index, velocity, max_vel, fps):
    before_event = np.nanmean(velocity[index - fps:index - 1])
    after_event = np.nanmean(velocity[index + 1:index + fps])

    if np.abs(before_event - after_event) < max_vel:
        return True
    return False


def get_event_end_index(event_start_index, hv_index, fps):
    hv_index_interval = np.diff(hv_index[event_start_index:])

    event_end_index = event_start_index
    for interval in hv_index_interval:
        if interval >= 2 * fps:
            break
        event_end_index += 1
    print("end: ", event_end_index)

    return event_end_index


def get_before_event(event_start_index, event_end_index, position, hv_index,
                     fps):
    event_start = hv_index[event_start_index]
    event_end = hv_index[event_end_index]
    if event_start_index != 0:
        last_event_interval = hv_index[event_start_index - 1]
    else:
        last_event_interval = 2 * fps

    if last_event_interval > fps:
        before_event = np.nanmean(position[event_start - fps:event_start + 1])
    elif 10 < last_event_interval < fps:
        before_event = np.nanmean(
            position[event_start - last_event_interval + 1:
                     event_start + 1])
    else:
        print(
            'Last event happened {} elements ago for event ({}, {})'.format(
                last_event_interval, event_start, event_end))
        before_event = np.nanmean(
            position[event_start - last_event_interval + 1:
                     event_start + 1])

    return before_event


def get_following_event(event_start_index, event_end_index, position,
                        max_index, hv_index, fps):
    event_start = hv_index[event_start_index]
    event_end = hv_index[event_end_index]

    if event_end_index < max_index - 2:
        following_event_interval = hv_index[event_end_index + 1]
    else:
        following_event_interval = 2 * fps

    if following_event_interval > fps:
        following_event_pos_mean = np.nanmean(position[event_end + 1:
                                                       event_end + fps])
    elif 10 < following_event_interval < fps:
        following_event_pos_mean = np.nanmean(
            position[event_end + 1:
                     event_end + following_event_interval - 1])
    else:
        print(
            'Following event is in {} elements for event ({}, {})'.format(
                following_event_interval, event_start, event_end))
        following_event_pos_mean = np.nanmean(
            position[event_end + 1:
                     event_end + following_event_interval - 1])

    return following_event_pos_mean
