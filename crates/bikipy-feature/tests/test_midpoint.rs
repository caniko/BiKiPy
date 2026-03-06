use polars::prelude::*;

use bikipy_feature::midpoint::midpoint_exprs;

#[test]
fn midpoint_of_two_points() {
    let df = df! {
        "x1" => &[0.0, 2.0],
        "y1" => &[0.0, 4.0],
        "x2" => &[4.0, 6.0],
        "y2" => &[4.0, 8.0],
    }
    .unwrap();

    let exprs = midpoint_exprs("x1", "y1", "x2", "y2", "mid_x", "mid_y");
    let mut lf = df.lazy();
    for expr in exprs {
        lf = lf.with_column(expr);
    }
    let result = lf.collect().unwrap();

    let mx = result.column("mid_x").unwrap().f64().unwrap();
    let my = result.column("mid_y").unwrap().f64().unwrap();

    assert!((mx.get(0).unwrap() - 2.0).abs() < 1e-10);
    assert!((my.get(0).unwrap() - 2.0).abs() < 1e-10);
    assert!((mx.get(1).unwrap() - 4.0).abs() < 1e-10);
    assert!((my.get(1).unwrap() - 6.0).abs() < 1e-10);
}

#[test]
fn midpoint_same_point() {
    let df = df! {
        "x1" => &[5.0],
        "y1" => &[5.0],
        "x2" => &[5.0],
        "y2" => &[5.0],
    }
    .unwrap();

    let exprs = midpoint_exprs("x1", "y1", "x2", "y2", "mx", "my");
    let mut lf = df.lazy();
    for expr in exprs {
        lf = lf.with_column(expr);
    }
    let result = lf.collect().unwrap();

    let mx = result.column("mx").unwrap().f64().unwrap();
    let my = result.column("my").unwrap().f64().unwrap();
    assert!((mx.get(0).unwrap() - 5.0).abs() < 1e-10);
    assert!((my.get(0).unwrap() - 5.0).abs() < 1e-10);
}
