use polars::prelude::*;

use bikipy_math::geometry::*;

#[test]
fn euclidean_distance_simple() {
    let df = df! {
        "x1" => &[0.0, 0.0],
        "y1" => &[0.0, 0.0],
        "x2" => &[3.0, 1.0],
        "y2" => &[4.0, 1.0],
    }
    .unwrap();

    let result = df
        .lazy()
        .with_column(euclidean_distance_expr("x1", "y1", "x2", "y2").alias("dist"))
        .collect()
        .unwrap();

    let dist = result.column("dist").unwrap().f64().unwrap();
    assert!((dist.get(0).unwrap() - 5.0).abs() < 1e-10);
    assert!((dist.get(1).unwrap() - std::f64::consts::SQRT_2).abs() < 1e-10);
}

#[test]
fn euclidean_distance_same_point() {
    let df = df! {
        "x1" => &[5.0],
        "y1" => &[5.0],
        "x2" => &[5.0],
        "y2" => &[5.0],
    }
    .unwrap();

    let result = df
        .lazy()
        .with_column(euclidean_distance_expr("x1", "y1", "x2", "y2").alias("dist"))
        .collect()
        .unwrap();

    let dist = result.column("dist").unwrap().f64().unwrap();
    assert!((dist.get(0).unwrap()).abs() < 1e-10);
}

#[test]
fn pixel_to_meters_scales() {
    let df = df! { "px" => &[100.0, 200.0, 0.0] }.unwrap();
    let result = df
        .lazy()
        .with_column(pixel_to_meters_expr("px", 0.001).alias("m"))
        .collect()
        .unwrap();

    let m = result.column("m").unwrap().f64().unwrap();
    assert!((m.get(0).unwrap() - 0.1).abs() < 1e-10);
    assert!((m.get(1).unwrap() - 0.2).abs() < 1e-10);
    assert!((m.get(2).unwrap() - 0.0).abs() < 1e-10);
}

#[test]
fn invert_y_flips_axis() {
    let df = df! { "y" => &[0.0, 50.0, 100.0] }.unwrap();
    let result = df
        .lazy()
        .with_column(invert_y_expr("y", 100.0).alias("y_inv"))
        .collect()
        .unwrap();

    let y_inv = result.column("y_inv").unwrap().f64().unwrap();
    assert!((y_inv.get(0).unwrap() - 100.0).abs() < 1e-10);
    assert!((y_inv.get(1).unwrap() - 50.0).abs() < 1e-10);
    assert!((y_inv.get(2).unwrap() - 0.0).abs() < 1e-10);
}

#[test]
fn perpendicular_distance_to_horizontal_line() {
    // Line from (0,0) to (10,0), points at various y-positions
    let df = df! {
        "px" => &[5.0, 5.0, 0.0],
        "py" => &[0.0, 3.0, -2.0],
    }
    .unwrap();

    let result = df
        .lazy()
        .with_column(
            perpendicular_distance_to_line_expr("px", "py", 0.0, 0.0, 10.0, 0.0).alias("dist"),
        )
        .collect()
        .unwrap();

    let dist = result.column("dist").unwrap().f64().unwrap();
    assert!((dist.get(0).unwrap() - 0.0).abs() < 1e-10); // On the line
    assert!((dist.get(1).unwrap() - 3.0).abs() < 1e-10); // 3 units above
    assert!((dist.get(2).unwrap() - 2.0).abs() < 1e-10); // 2 units below (abs)
}

#[test]
fn signed_angle_reference() {
    let df = df! {
        "dx1" => &[1.0, 0.0],
        "dy1" => &[0.0, 1.0],
        "dx2" => &[0.0, 1.0],
        "dy2" => &[1.0, 0.0],
    }
    .unwrap();

    let result = df
        .lazy()
        .with_column(signed_angle_expr("dx1", "dy1", "dx2", "dy2").alias("angle"))
        .collect()
        .unwrap();

    let angle = result.column("angle").unwrap().f64().unwrap();
    // (1,0) to (0,1) → +π/2
    assert!((angle.get(0).unwrap() - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
    // (0,1) to (1,0) → -π/2
    assert!((angle.get(1).unwrap() - (-std::f64::consts::FRAC_PI_2)).abs() < 1e-10);
}
