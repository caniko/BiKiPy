use bikipy_math::confinement::*;

#[test]
fn point_in_square() {
    let square = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
    assert!(point_in_polygon(0.5, 0.5, &square));
    assert!(!point_in_polygon(1.5, 0.5, &square));
}

#[test]
fn point_in_triangle() {
    let tri = vec![(0.0, 0.0), (2.0, 0.0), (1.0, 2.0)];
    assert!(point_in_polygon(1.0, 0.5, &tri));
    assert!(!point_in_polygon(0.0, 2.0, &tri));
}

#[test]
fn point_in_polygon_degenerate() {
    // Fewer than 3 vertices
    assert!(!point_in_polygon(0.0, 0.0, &[]));
    assert!(!point_in_polygon(0.0, 0.0, &[(0.0, 0.0), (1.0, 0.0)]));
}

#[test]
fn point_in_ellipse_center() {
    assert!(point_in_ellipse(0.0, 0.0, 0.0, 0.0, 1.0, 1.0));
}

#[test]
fn point_in_ellipse_on_boundary() {
    assert!(point_in_ellipse(1.0, 0.0, 0.0, 0.0, 1.0, 1.0));
}

#[test]
fn point_outside_ellipse() {
    assert!(!point_in_ellipse(2.0, 0.0, 0.0, 0.0, 1.0, 1.0));
}

#[test]
fn point_in_ellipse_non_circular() {
    // Semi-axes: rx=2, ry=1
    assert!(point_in_ellipse(1.5, 0.0, 0.0, 0.0, 2.0, 1.0));
    assert!(!point_in_ellipse(0.0, 1.5, 0.0, 0.0, 2.0, 1.0));
}

#[test]
fn point_in_ellipse_offset_center() {
    // Ellipse centered at (5, 5) with radius 1
    assert!(point_in_ellipse(5.0, 5.0, 5.0, 5.0, 1.0, 1.0));
    assert!(point_in_ellipse(5.5, 5.0, 5.0, 5.0, 1.0, 1.0));
    assert!(!point_in_ellipse(7.0, 5.0, 5.0, 5.0, 1.0, 1.0));
}
