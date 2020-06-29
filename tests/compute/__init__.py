from pathlib import Path

from kinpy.compute import angle_over_time
import kinpy


HDF_FILE = Path().resolve() / "data_for_angle.h5"


def test_angle():
    df = DeepLabCutReader.from_hdf(
        HDF_FILE,
        video_res=(1280, 720),
        center_bp=[("left_ear", "right_ear")],
        future_scaling=False,
    )

    tail_base_tip_angle = angle_over_time(
        df, point_a="tail_base", point_b="tail_mid", point_c="tail_tip"
    )

    tail_base_tip_angle = angle_over_time(
        df, point_a="tail_base", point_b="tail_mid", point_c="tail_tip"
    )

    head_tail = angle_over_time(
        df, point_a="c_left_ear_right_ear", point_b="tail_base", point_c="tail_tip"
    )
