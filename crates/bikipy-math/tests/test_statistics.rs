use polars::prelude::*;

use bikipy_math::statistics::*;

#[test]
fn interpolate_fills_nulls_linearly() {
    let df = df! {
        "v" => &[Some(0.0), None, Some(2.0), None, Some(4.0)],
    }
    .unwrap();

    let result = df
        .lazy()
        .with_column(interpolate_expr("v").alias("v"))
        .collect()
        .unwrap();

    let v = result.column("v").unwrap().f64().unwrap();
    assert!((v.get(1).unwrap() - 1.0).abs() < 1e-10);
    assert!((v.get(3).unwrap() - 3.0).abs() < 1e-10);
}

#[test]
fn forward_fill_carries_last_value() {
    let df = df! {
        "v" => &[Some(1.0), None, None, Some(5.0), None],
    }
    .unwrap();

    let result = df
        .lazy()
        .with_column(forward_fill_expr("v").alias("v"))
        .collect()
        .unwrap();

    let v = result.column("v").unwrap().f64().unwrap();
    assert!((v.get(1).unwrap() - 1.0).abs() < 1e-10);
    assert!((v.get(2).unwrap() - 1.0).abs() < 1e-10);
    assert!((v.get(4).unwrap() - 5.0).abs() < 1e-10);
}

#[test]
fn median_filter_smooths() {
    let df = df! {
        "v" => &[1.0, 100.0, 1.0, 1.0, 1.0],
    }
    .unwrap();

    let result = df
        .lazy()
        .with_column(median_filter_expr("v", 3).alias("smooth"))
        .collect()
        .unwrap();

    let smooth = result.column("smooth").unwrap().f64().unwrap();
    // The spike at index 1 should be dampened by the median filter
    // window [1, 100, 1] → median = 1
    assert!((smooth.get(1).unwrap() - 1.0).abs() < 1e-10);
}
