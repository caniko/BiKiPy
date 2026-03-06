use polars::prelude::*;

use bikipy_core::types::CoordinateColumns;
use bikipy_feature::heuristic::Heuristic;
use bikipy_feature::heuristic::solo::body_proximity::BodyProximityHeuristic;
use bikipy_feature::axiom::proximity::ComputeProximity;
use bikipy_perimeter::circle::Circle;
use bikipy_perimeter::perimeter::Perimeter;

fn make_test_df() -> DataFrame {
    // Object at (5, 5). Animal body parts at various distances.
    df! {
        "center_ear_x" => &[5.0, 0.0, 5.05],
        "center_ear_y" => &[5.0, 0.0, 5.0],
        "tail_base_x" => &[4.0, -1.0, 4.0],
        "tail_base_y" => &[4.0, -1.0, 4.0],
    }
    .unwrap()
}

#[test]
fn body_proximity_heuristic_evaluates() {
    let perimeter = Perimeter::new("obj", Circle::new(5.0, 5.0, 0.5));
    let h = BodyProximityHeuristic {
        perimeter,
        max_distance: 0.1,
    };

    assert_eq!(h.name(), "body_proximity");

    let result = h.evaluate(make_test_df().lazy()).collect().unwrap();
    assert!(result.column("body_proximity").is_ok());
    let bp = result.column("body_proximity").unwrap().bool().unwrap();
    assert_eq!(bp.len(), 3);
}

#[test]
fn heuristic_summary_default() {
    let perimeter = Perimeter::new("obj", Circle::new(5.0, 5.0, 0.5));
    let h = BodyProximityHeuristic {
        perimeter,
        max_distance: 2.0, // large distance so some points are inside
    };

    let df = h.evaluate(make_test_df().lazy()).collect().unwrap();
    let summary = h.summary(&df, 30.0);
    assert_eq!(summary.name, "body_proximity");
    assert_eq!(summary.total_frames, 3);
    assert!(summary.seconds >= 0.0);
}

#[test]
fn compute_proximity_axiom() {
    let perimeter = Perimeter::new("obj", Circle::new(5.0, 5.0, 0.5));
    let proximity = ComputeProximity::new(perimeter, 1.0);
    let coords = CoordinateColumns::new("center_ear_x", "center_ear_y");

    let result = proximity
        .apply(make_test_df().lazy(), &coords, "near_obj")
        .collect()
        .unwrap();

    let near = result.column("near_obj").unwrap().bool().unwrap();
    assert_eq!(near.len(), 3);
    // Point at (5,5) should be near the object at (5,5) with radius 0.5 + distance 1.0
    assert_eq!(near.get(0).unwrap(), true);
    // Point at (0,0) should not be near
    assert_eq!(near.get(1).unwrap(), false);
}
