from pydantic import FilePath


def get_file_label_from_2nd_str_in_split(meters_per_pixel_file_path: FilePath):
    return meters_per_pixel_file_path.stem.split("-")[1]


def get_file_label_from_3rd_str_in_split(meters_per_pixel_file_path: FilePath):
    return meters_per_pixel_file_path.stem.split("-")[2]
