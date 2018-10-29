import numpy as np


def get_velocity(df, body_part):
    x = df.loc[:, (body_part, 'x')].interpolate(method='linear').values
    y = df.loc[:, (body_part, 'y')].interpolate(method='linear').values
    return np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)

def get_position(df, body_part):
    x = df.loc[:, (body_part, 'x')].interpolate(method='linear').values
    y = df.loc[:, (body_part, 'y')].interpolate(method='linear').values
    return np.sqrt(x ** 2 + y ** 2)

def high_velocity_values(df, body_part, max_vel, range_thresh, fps=30):
    velocity = get_velocity(df, body_part)
    hv_values = np.zeros(len(velocity) + 1, dtype=bool)

    bad_velocity_lcs = np.where(velocity > max_vel)[0]
    if not bad_velocity_lcs.size:
        return hv_values

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
        if g_len == 0:
            pass
        elif g_len == 1:
            before_event = np.mean(
                velocity[group[0] - 10: group[0]])
            after_event = np.mean(
                velocity[group[0] + fps: group[0] + fps * 2 + 1])

            if abs(after_event - before_event) < max_vel:
                hv_values[group[0]:group[0] + fps * 2 + 1] = True

        elif g_len == 2:
            hv_values[group[0]:group[1] + 1] = True

        else:
            print(group)
            close_hv = np.abs(np.diff(group) - 1)
            i = 0
            while i < g_len - 2:
                if close_hv[i] <= 1:
                    print(group[i])
                    hv_values[group[i]:group[i + 1] + 1] = True
                    i += 2
                    continue

                if close_hv[i - 1] <= 1:
                    print(group[i])
                    hv_values[group[i - 1]:group[i] + 1] = True
                    print('skipped', group[i], group[i + 1])
                    i += 1
                    continue

                if i > 0:
                    print(group[i])
                    before_event = np.mean(
                        velocity[group[i] - close_hv[i - 1]:group[i]])

                elif i == 0:
                    print(group[i])
                    before_event = np.mean(
                        velocity[group[i] - fps:group[i] + 1])

                if i == g_len - 2:
                    after_event = np.mean(
                        velocity[group[i + 1] + 1:group[i + 1] + fps + 1])

                elif close_hv[i + 1] <= 1:
                    print(group[i])
                    print(close_hv[i])
                    i += 1
                    continue

                else:
                    print(group[i])
                    after_event = np.mean(
                        velocity[
                        group[i + 1] + 1:group[i + 1] - close_hv[i + 1] + 1])

                event_average = np.mean(
                    velocity[group[i]:group[i + 1] + 1])

                if before_event < event_average < after_event or before_event > event_average > after_event:
                    print(group[i])
                    for p in range(group[i], group[i + 1] + 1):
                        if not before_event < velocity[
                            p] < after_event or not before_event > velocity[
                            p] > after_event:
                            print('pass')
                            hv_values[p] = True
                    i += 2
                else:
                    i += 1

                print(before_event)
                print(event_average)
                print(after_event)

            if i == g_len - 1:
                before_event = np.mean(
                    velocity[group[i] - fps:group[i]])
                after_event = np.mean(
                    velocity[group[i] + fps:group[i] + fps * 2 + 1])

                if abs(after_event - before_event) < max_vel:
                    hv_values[group[i]:group[i] + fps * 2 + 1] = True

        print(np.where(hv_values is True))
        return hv_values


def invalid_relative_part_distance(df, part_a, part_b, max_dist):
    pos_a = get_position(df, part_a)
    pos_b = get_position(df, part_b)
    if pos_a.length != pos_b.length:
        msg = 'Array a and b have different lengths,' \
              '{} and {}'.format(pos_a.length, pos_b.length)
        raise ValueError(msg)

    dist_ab = np.abs(pos_a - pos_b)
    rpd_values = np.zeros(len(pos_a) + 1, dtype=bool)
    rpd_values[np.where(dist_ab > max_dist)] = True

    return rpd_values
