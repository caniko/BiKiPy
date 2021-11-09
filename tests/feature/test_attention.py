from bikipy.feature.attention import proximity_filter

from tests.test_data.perimeter.get_perimeter import perimeter_object


def test_proximity_filter():
    proximity_filter(
        perimeter=perimeter_object,
        nose: Sequence[Sequence[float]],
        center_eye: Sequence[Sequence[float]],
        perimeter_border_normal_pixel_magnitude: float,
    )
