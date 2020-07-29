

def find_first_valid_index(valid_point_locations, trim_tolerance, max):
    first_valid_point = max
    for point_name, data in valid_point_locations.items():
        i = 0
        while i < data.size:
            if data[i] > first_valid_point:
                break
            for increment in range(i, i+trim_tolerance):
                if data[i+increment] + 1 != data[i+increment+1]:
                    break
            else:
                if first_valid_point > data[i]:
                    first_valid_point = data[i]
                    break
            i += 1

    return first_valid_point
