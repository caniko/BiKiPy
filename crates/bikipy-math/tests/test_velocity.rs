use polars::prelude::*;

use bikipy_math::velocity::*;

#[test]
fn velocity_computes_displacement() {
    // Points at (0,0), (3,4), (3,4) — speeds: null, 5.0, 0.0
    let df = df! {
        "x" => &[0.0, 3.0, 3.0],
        "y" => &[0.0, 4.0, 4.0],
    }
    .unwrap();

    let result = df
        .lazy()
        .with_column(velocity_expr("x", "y").alias("speed"))
        .collect()
        .unwrap();

    let speed = result.column("speed").unwrap().f64().unwrap();
    assert!(speed.get(0).is_none()); // first frame has no previous
    assert!((speed.get(1).unwrap() - 5.0).abs() < 1e-10);
    assert!((speed.get(2).unwrap() - 0.0).abs() < 1e-10);
}

#[test]
fn high_velocity_mask_flags_teleportation() {
    let df = df! {
        "x" => &[0.0, 100.0, 101.0],
        "y" => &[0.0, 0.0, 0.0],
    }
    .unwrap();

    let result = df
        .lazy()
        .with_column(high_velocity_mask_expr("x", "y", 50.0).alias("high"))
        .collect()
        .unwrap();

    let high = result.column("high").unwrap().bool().unwrap();
    // Frame 0: null (no previous), Frame 1: speed=100 > 50 → true, Frame 2: speed=1 ≤ 50 → false
    assert!(high.get(0).is_none());
    assert!(high.get(1).unwrap());
    assert!(!high.get(2).unwrap());
}

#[test]
fn null_high_velocity_nullifies_coordinates() {
    let df = df! {
        "x" => &[0.0, 100.0, 101.0],
        "y" => &[0.0, 0.0, 0.0],
    }
    .unwrap();

    // Apply both x and y expressions simultaneously to avoid
    // sequential column mutation affecting the velocity mask.
    let exprs = null_high_velocity_exprs("x", "y", 50.0);
    let result = df.lazy().with_columns(exprs).collect().unwrap();

    let x = result.column("x").unwrap().f64().unwrap();
    let y = result.column("y").unwrap().f64().unwrap();

    // Frame 1: velocity=100 > 50 → nullified
    assert!(x.get(1).is_none());
    assert!(y.get(1).is_none());
    // Frame 2: velocity=1 ≤ 50 → keep original
    assert!((x.get(2).unwrap() - 101.0).abs() < 1e-10);
    assert!((y.get(2).unwrap() - 0.0).abs() < 1e-10);
}
