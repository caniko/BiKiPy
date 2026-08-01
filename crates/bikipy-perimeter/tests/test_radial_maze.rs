use bikipy_core::shape::{RayIntersectable, Shape};
use bikipy_perimeter::polygon::Polygon;
use bikipy_perimeter::radial_maze::RadialMaze;

fn sample_maze() -> RadialMaze {
    let center = Polygon::new(vec![(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]);
    let arm1 = Polygon::new(vec![(1.0, -0.5), (4.0, -0.5), (4.0, 0.5), (1.0, 0.5)]);
    let arm2 = Polygon::new(vec![(-0.5, 1.0), (0.5, 1.0), (0.5, 4.0), (-0.5, 4.0)]);
    RadialMaze::new(center, vec![arm1, arm2])
}

#[test]
fn maze_contains_center() {
    let m = sample_maze();
    assert!(m.contains(0.0, 0.0));
    assert!(m.in_center(0.0, 0.0));
}

#[test]
fn maze_contains_arm() {
    let m = sample_maze();
    assert!(m.contains(2.0, 0.0)); // in arm1
    assert!(m.contains(0.0, 2.0)); // in arm2
}

#[test]
fn maze_excludes_outside() {
    let m = sample_maze();
    assert!(!m.contains(5.0, 5.0));
}

#[test]
fn maze_arm_containing() {
    let m = sample_maze();
    assert_eq!(m.arm_containing(2.0, 0.0), Some(0));
    assert_eq!(m.arm_containing(0.0, 2.0), Some(1));
    assert_eq!(m.arm_containing(0.0, 0.0), None); // center, not an arm
}

#[test]
fn maze_in_center() {
    let m = sample_maze();
    assert!(m.in_center(0.0, 0.0));
    assert!(!m.in_center(2.0, 0.0));
}

#[test]
fn maze_bounding_box() {
    let m = sample_maze();
    let bb = m.bounding_box();
    assert!((bb.min_x - (-1.0)).abs() < 1e-10);
    assert!((bb.min_y - (-1.0)).abs() < 1e-10);
    assert!((bb.max_x - 4.0).abs() < 1e-10);
    assert!((bb.max_y - 4.0).abs() < 1e-10);
}

#[test]
fn maze_ray_intersects_arm() {
    let m = sample_maze();
    // Ray from far left pointing right should hit center + arm1
    assert!(m.ray_intersects((-5.0, 0.0), (1.0, 0.0)));
}

#[test]
fn maze_ray_misses() {
    let m = sample_maze();
    assert!(!m.ray_intersects((-5.0, 10.0), (1.0, 0.0)));
}
