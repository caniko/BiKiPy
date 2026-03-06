use polars::prelude::*;

use bikipy_feature::angle::add_direction_columns;

#[test]
fn add_direction_columns_computes_diff() {
    let df = df! {
        "from_x" => &[0.0, 1.0],
        "from_y" => &[0.0, 2.0],
        "to_x" => &[3.0, 4.0],
        "to_y" => &[4.0, 6.0],
    }
    .unwrap();

    let result = add_direction_columns(df.lazy(), "from_x", "from_y", "to_x", "to_y", "heading")
        .collect()
        .unwrap();

    let dx = result.column("heading_dx").unwrap().f64().unwrap();
    let dy = result.column("heading_dy").unwrap().f64().unwrap();

    assert!((dx.get(0).unwrap() - 3.0).abs() < 1e-10);
    assert!((dy.get(0).unwrap() - 4.0).abs() < 1e-10);
    assert!((dx.get(1).unwrap() - 3.0).abs() < 1e-10);
    assert!((dy.get(1).unwrap() - 4.0).abs() < 1e-10);
}
