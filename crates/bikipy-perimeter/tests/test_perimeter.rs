use bikipy_core::types::CoordinateColumns;
use bikipy_perimeter::circle::Circle;
use bikipy_perimeter::perimeter::Perimeter;
use polars::prelude::*;

#[test]
fn perimeter_new() {
    let p = Perimeter::new("object1", Circle::new(0.0, 0.0, 1.0));
    assert_eq!(p.label, "object1");
}

#[test]
fn perimeter_confinement_mask() {
    let p = Perimeter::new("obj", Circle::new(5.0, 5.0, 2.0));
    let coords = CoordinateColumns::new("x", "y");

    let df = df! {
        "x" => &[5.0, 0.0, 6.0],
        "y" => &[5.0, 0.0, 5.0],
    }
    .unwrap();

    let result = df
        .lazy()
        .with_column(p.confinement_mask(&coords).alias("inside"))
        .collect()
        .unwrap();

    let inside = result.column("inside").unwrap().bool().unwrap();
    assert_eq!(inside.get(0).unwrap(), true); // center
    assert_eq!(inside.get(1).unwrap(), false); // far away
    assert_eq!(inside.get(2).unwrap(), true); // within radius
}

#[test]
fn perimeter_filter_confined() {
    let p = Perimeter::new("obj", Circle::new(0.0, 0.0, 1.0));
    let coords = CoordinateColumns::new("x", "y");

    let df = df! {
        "x" => &[0.0, 5.0, 0.5],
        "y" => &[0.0, 5.0, 0.5],
    }
    .unwrap();

    let result = p.filter_confined(df.lazy(), &coords).collect().unwrap();
    assert_eq!(result.height(), 2); // only 2 points inside
}

#[test]
fn perimeter_expanded() {
    let p = Perimeter::new("obj", Circle::new(0.0, 0.0, 1.0));
    let expanded = p.expanded(0.5);
    assert_eq!(expanded.label, "obj_expanded_0.5");
    assert!((expanded.shape.radius - 1.5).abs() < 1e-10);
}
