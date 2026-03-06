use bikipy_core::shape::{Expandable, RayIntersectable, Shape};
use bikipy_perimeter::rectangle::Rectangle;

#[test]
fn rectangle_contains_center() {
    let r = Rectangle::new(5.0, 5.0, 4.0, 2.0);
    assert!(r.contains(5.0, 5.0));
}

#[test]
fn rectangle_contains_edge() {
    let r = Rectangle::new(5.0, 5.0, 4.0, 2.0);
    // Edge at (3, 4) to (7, 6)
    assert!(r.contains(3.5, 5.0));
}

#[test]
fn rectangle_excludes_outside() {
    let r = Rectangle::new(5.0, 5.0, 4.0, 2.0);
    assert!(!r.contains(0.0, 0.0));
    assert!(!r.contains(8.0, 5.0));
}

#[test]
fn rectangle_bounding_box() {
    let r = Rectangle::new(5.0, 5.0, 4.0, 2.0);
    let bb = r.bounding_box();
    assert!((bb.min_x - 3.0).abs() < 1e-10);
    assert!((bb.min_y - 4.0).abs() < 1e-10);
    assert!((bb.max_x - 7.0).abs() < 1e-10);
    assert!((bb.max_y - 6.0).abs() < 1e-10);
}

#[test]
fn rectangle_expand_grows() {
    let r = Rectangle::new(5.0, 5.0, 4.0, 2.0);
    let expanded = r.expand(1.0);
    // Expanded polygon should contain points outside the original
    assert!(expanded.contains(2.5, 5.0));
    assert!(!r.contains(2.5, 5.0));
}

#[test]
fn rectangle_ray_hit() {
    let r = Rectangle::new(5.0, 5.0, 4.0, 2.0);
    // Ray from origin pointing right should hit
    assert!(r.ray_intersects((0.0, 5.0), (1.0, 0.0)));
}

#[test]
fn rectangle_ray_miss() {
    let r = Rectangle::new(5.0, 5.0, 4.0, 2.0);
    // Ray pointing up should miss
    assert!(!r.ray_intersects((0.0, 0.0), (0.0, -1.0)));
}

#[test]
fn rectangle_serde_roundtrip() {
    let r = Rectangle::new(1.0, 2.0, 3.0, 4.0);
    let json = serde_json::to_string(&r).unwrap();
    let r2: Rectangle = serde_json::from_str(&json).unwrap();
    assert_eq!(r, r2);
}
