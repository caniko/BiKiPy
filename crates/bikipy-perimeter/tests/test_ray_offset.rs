use bikipy_perimeter::circle::Circle;
use bikipy_perimeter::ray_offset::RayOffsetFilter;

#[test]
fn ray_offset_filter_new() {
    let filter = RayOffsetFilter::new(Circle::new(5.0, 0.0, 1.0), 45.0);
    assert!((filter.max_angle_rad - std::f64::consts::FRAC_PI_4).abs() < 1e-10);
}

#[test]
fn ray_offset_filter_stores_shape() {
    let c = Circle::new(10.0, 10.0, 2.0);
    let filter = RayOffsetFilter::new(c, 30.0);
    assert!((filter.shape.center_x - 10.0).abs() < 1e-10);
    assert!((filter.shape.radius - 2.0).abs() < 1e-10);
}
