use polars::prelude::*;

use bikipy_feature::motion::*;

#[test]
fn add_speed_column_computes_velocity() {
    let df = df! {
        "x" => &[0.0, 3.0, 3.0],
        "y" => &[0.0, 4.0, 4.0],
    }
    .unwrap();

    let result = add_speed_column(df.lazy(), "x", "y", "speed")
        .collect()
        .unwrap();

    let speed = result.column("speed").unwrap().f64().unwrap();
    assert!(speed.get(0).is_none()); // first frame
    assert!((speed.get(1).unwrap() - 5.0).abs() < 1e-10);
    assert!((speed.get(2).unwrap() - 0.0).abs() < 1e-10);
}

#[test]
fn displacement_from_origin() {
    let df = df! {
        "x" => &[0.0, 3.0, 0.0],
        "y" => &[0.0, 4.0, 0.0],
    }
    .unwrap();

    let result = df
        .lazy()
        .with_column(displacement_expr("x", "y").alias("disp"))
        .collect()
        .unwrap();

    let disp = result.column("disp").unwrap().f64().unwrap();
    assert!((disp.get(0).unwrap() - 0.0).abs() < 1e-10); // at origin
    assert!((disp.get(1).unwrap() - 5.0).abs() < 1e-10); // distance from (0,0) to (3,4)
    assert!((disp.get(2).unwrap() - 0.0).abs() < 1e-10); // back to origin
}
