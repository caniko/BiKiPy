use polars::prelude::*;

use bikipy_math::vector::*;

fn make_df(x: &[f64], y: &[f64]) -> DataFrame {
    df! {
        "x" => x,
        "y" => y,
    }
    .unwrap()
}

#[test]
fn unit_vector_normalizes() {
    let df = make_df(&[3.0, 0.0], &[4.0, 1.0]);
    let lf = df.lazy().with_columns(unit_vector_exprs("x", "y", "v"));
    let result = lf.collect().unwrap();

    let ux = result.column("v_ux").unwrap().f64().unwrap();
    let uy = result.column("v_uy").unwrap().f64().unwrap();

    // (3, 4) → magnitude 5 → (0.6, 0.8)
    assert!((ux.get(0).unwrap() - 0.6).abs() < 1e-10);
    assert!((uy.get(0).unwrap() - 0.8).abs() < 1e-10);

    // (0, 1) → magnitude 1 → (0, 1)
    assert!((ux.get(1).unwrap() - 0.0).abs() < 1e-10);
    assert!((uy.get(1).unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn orthogonal_vector_perpendicular() {
    let df = make_df(&[1.0, 0.0], &[0.0, 1.0]);
    let lf = df
        .lazy()
        .with_columns(orthogonal_vector_exprs("x", "y", "o"));
    let result = lf.collect().unwrap();

    let ox = result.column("o_ortho_x").unwrap().f64().unwrap();
    let oy = result.column("o_ortho_y").unwrap().f64().unwrap();

    // (1, 0) → ortho = (0, 1)
    assert!((ox.get(0).unwrap() - 0.0).abs() < 1e-10);
    assert!((oy.get(0).unwrap() - 1.0).abs() < 1e-10);

    // (0, 1) → ortho = (-1, 0)
    assert!((ox.get(1).unwrap() - (-1.0)).abs() < 1e-10);
    assert!((oy.get(1).unwrap() - 0.0).abs() < 1e-10);
}

#[test]
fn dot_product_correct() {
    let df = df! {
        "a_x" => &[1.0, 2.0],
        "a_y" => &[0.0, 3.0],
        "b_x" => &[0.0, 4.0],
        "b_y" => &[1.0, 5.0],
    }
    .unwrap();

    let result = df
        .lazy()
        .with_column(dot_product_expr("a_x", "a_y", "b_x", "b_y").alias("dot"))
        .collect()
        .unwrap();

    let dot = result.column("dot").unwrap().f64().unwrap();
    // (1,0)·(0,1) = 0
    assert!((dot.get(0).unwrap() - 0.0).abs() < 1e-10);
    // (2,3)·(4,5) = 8+15 = 23
    assert!((dot.get(1).unwrap() - 23.0).abs() < 1e-10);
}

#[test]
fn direction_exprs_computes_diff() {
    let df = df! {
        "ax" => &[0.0, 1.0],
        "ay" => &[0.0, 2.0],
        "bx" => &[3.0, 4.0],
        "by" => &[4.0, 6.0],
    }
    .unwrap();

    let result = df
        .lazy()
        .with_columns(direction_exprs("ax", "ay", "bx", "by", "d"))
        .collect()
        .unwrap();

    let dx = result.column("d_dx").unwrap().f64().unwrap();
    let dy = result.column("d_dy").unwrap().f64().unwrap();

    assert!((dx.get(0).unwrap() - 3.0).abs() < 1e-10);
    assert!((dy.get(0).unwrap() - 4.0).abs() < 1e-10);
    assert!((dx.get(1).unwrap() - 3.0).abs() < 1e-10);
    assert!((dy.get(1).unwrap() - 4.0).abs() < 1e-10);
}

#[test]
fn rotate_90_degrees() {
    let df = make_df(&[1.0], &[0.0]);
    let angle = std::f64::consts::FRAC_PI_2;
    let result = df
        .lazy()
        .with_columns(rotate_exprs("x", "y", angle, "r"))
        .collect()
        .unwrap();

    let rx = result.column("r_rot_x").unwrap().f64().unwrap();
    let ry = result.column("r_rot_y").unwrap().f64().unwrap();

    // Rotating (1,0) by 90° → (0,1)
    assert!(rx.get(0).unwrap().abs() < 1e-10);
    assert!((ry.get(0).unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn ray_hits_segment() {
    assert!(ray_line_segment_intersection(
        (0.0, 0.0),
        (1.0, 0.0),
        (5.0, -1.0),
        (5.0, 1.0),
    ));
}

#[test]
fn ray_misses_segment() {
    assert!(!ray_line_segment_intersection(
        (0.0, 0.0),
        (0.0, 1.0),
        (5.0, -1.0),
        (5.0, 1.0),
    ));
}

#[test]
fn ray_behind_origin() {
    // Segment is behind the ray origin
    assert!(!ray_line_segment_intersection(
        (0.0, 0.0),
        (1.0, 0.0),
        (-5.0, -1.0),
        (-5.0, 1.0),
    ));
}

#[test]
fn ray_parallel_to_segment() {
    assert!(!ray_line_segment_intersection(
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (5.0, 1.0),
    ));
}

#[test]
fn angle_between_same_direction() {
    let angle = angle_between((1.0, 0.0), (1.0, 0.0));
    assert!(angle.abs() < 1e-10);
}

#[test]
fn angle_between_orthogonal() {
    let angle = angle_between((1.0, 0.0), (0.0, 1.0));
    assert!((angle - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
}

#[test]
fn angle_between_opposite() {
    let angle = angle_between((1.0, 0.0), (-1.0, 0.0));
    assert!((angle - std::f64::consts::PI).abs() < 1e-10);
}
