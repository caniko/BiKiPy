use bikipy_core::shape::{Expandable, RayIntersectable, Shape};
use bikipy_perimeter::triangle::Triangle;

#[test]
fn triangle_contains_inside() {
    let t = Triangle::new((0.0, 0.0), (4.0, 0.0), (2.0, 3.0));
    assert!(t.contains(2.0, 1.0));
}

#[test]
fn triangle_excludes_outside() {
    let t = Triangle::new((0.0, 0.0), (4.0, 0.0), (2.0, 3.0));
    assert!(!t.contains(5.0, 5.0));
    assert!(!t.contains(-1.0, 0.0));
}

#[test]
fn triangle_vertices() {
    let t = Triangle::new((0.0, 0.0), (1.0, 0.0), (0.5, 1.0));
    assert_eq!(t.vertices().len(), 3);
    assert_eq!(t.vertices()[0], (0.0, 0.0));
}

#[test]
fn triangle_bounding_box() {
    let t = Triangle::new((1.0, 2.0), (5.0, 2.0), (3.0, 6.0));
    let bb = t.bounding_box();
    assert!((bb.min_x - 1.0).abs() < 1e-10);
    assert!((bb.min_y - 2.0).abs() < 1e-10);
    assert!((bb.max_x - 5.0).abs() < 1e-10);
    assert!((bb.max_y - 6.0).abs() < 1e-10);
}

#[test]
fn triangle_expand() {
    let t = Triangle::new((0.0, 0.0), (4.0, 0.0), (2.0, 3.0));
    let expanded = t.expand(0.5);
    // Expanded should contain points just outside original
    assert!(expanded.contains(2.0, 1.0)); // still inside
}

#[test]
fn triangle_ray_hit() {
    let t = Triangle::new((5.0, -1.0), (5.0, 1.0), (7.0, 0.0));
    assert!(t.ray_intersects((0.0, 0.0), (1.0, 0.0)));
}

#[test]
fn triangle_ray_miss() {
    let t = Triangle::new((5.0, 5.0), (7.0, 5.0), (6.0, 7.0));
    assert!(!t.ray_intersects((0.0, 0.0), (1.0, 0.0)));
}

#[test]
fn triangle_serde_roundtrip() {
    let t = Triangle::new((0.0, 0.0), (1.0, 0.0), (0.5, 1.0));
    let json = serde_json::to_string(&t).unwrap();
    let t2: Triangle = serde_json::from_str(&json).unwrap();
    assert_eq!(t, t2);
}
