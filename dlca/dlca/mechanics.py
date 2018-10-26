import numpy as np


def high_velocity_values(df, body_part, max_vel, range_thresh):
    x = df.loc[:, (body_part, 'x')].values
    y = df.loc[:, (body_part, 'y')].values
    hv_values = np.zeros(len(x), dtype=bool)

    velocity = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    bad_velocity_lcs = np.where(velocity > max_vel)[0]
    bad_lcs_intervals = np.diff(bad_velocity_lcs)

    bad_velocity_ranges = []
    group_index = -1
    keep = False
    for i, interval in enumerate(bad_lcs_intervals):
        if interval < range_thresh:
            if not keep:
                keep = True
                bad_velocity_ranges.append([bad_velocity_lcs[i] + 1])
                group_index += 1
            else:
                bad_velocity_ranges[group_index].\
                    append(bad_velocity_lcs[i] + 1)
        else:
            if keep:
                bad_velocity_ranges[group_index].append(
                    bad_velocity_lcs[i] + 1)
                keep = False
            else:
                bad_velocity_ranges.append([bad_velocity_lcs[i] + 1])
                group_index += 1

    for group in bad_velocity_ranges:
        g_len = len(group)
        if g_len == 1:
            hv_values[group[0]] = True

        elif g_len == 2:
            hv_values[range(group[0], group[1])] = True

        elif np.isclose(g_len % 2, 0):
            for i in range(0, g_len, 2):

                before_event = np.average(
                    velocity[group[i] - 10: group[i] - 3])
                after_event = np.average(
                    velocity[group[i+1] + 3: group[i+1] + 10])
                event_average = np.average(
                    hv_values[range(group[i], group[i + 1])])

                if not before_event < event_average < after_event or (
                        not before_event > event_average > after_event):
                    hv_values[range(group[i], group[i + 1])] = True

                else:
                    for p in range(group[i], group[i + 1]):
                        if not before_event < velocity[p] < after_event or (
                                not before_event > velocity[p] > after_event):
                            hv_values[p] = True

        elif np.isclose(g_len % 3, 0):
            pass

    return hv_values
