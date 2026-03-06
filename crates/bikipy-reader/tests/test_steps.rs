use polars::prelude::*;

use bikipy_reader::pipeline::PipelineStep;
use bikipy_reader::steps::likelihood::LikelihoodFilter;
use bikipy_reader::steps::midpoint::MidpointComputer;
use bikipy_reader::steps::smoothing::MedianSmoother;
use bikipy_reader::steps::transform::CoordinateTransform;
use bikipy_reader::steps::velocity::HighVelocityFilter;
use bikipy_core::types::MidpointGroup;

#[test]
fn likelihood_filter_nulls_low_confidence() {
    let df = df! {
        "nose_x" => &[1.0, 2.0, 3.0],
        "nose_likelihood" => &[0.9, 0.1, 0.8],
    }
    .unwrap();

    let step = LikelihoodFilter::new(
        0.5,
        vec!["nose_likelihood".into()],
        vec!["nose_x".into()],
    );
    assert_eq!(step.name(), "likelihood_filter");

    let result = step.apply(df.lazy()).collect().unwrap();
    let x = result.column("nose_x").unwrap().f64().unwrap();
    assert!((x.get(0).unwrap() - 1.0).abs() < 1e-10);
    assert!(x.get(1).is_none()); // below threshold → null
    assert!((x.get(2).unwrap() - 3.0).abs() < 1e-10);
}

#[test]
fn high_velocity_filter_nulls_teleports() {
    let df = df! {
        "x" => &[0.0, 100.0, 101.0, 102.0],
        "y" => &[0.0, 0.0, 0.0, 0.0],
    }
    .unwrap();

    let step = HighVelocityFilter {
        max_velocity: 50.0,
        x_columns: vec!["x".into()],
        y_columns: vec!["y".into()],
    };
    assert_eq!(step.name(), "high_velocity_filter");

    let result = step.apply(df.lazy()).collect().unwrap();
    let x = result.column("x").unwrap().f64().unwrap();
    // Frame 1 had velocity 100 > 50 → nullified
    assert!(x.get(1).is_none());
}

#[test]
fn coordinate_transform_pixels_to_meters() {
    let df = df! {
        "x" => &[100.0, 200.0],
        "y" => &[50.0, 100.0],
    }
    .unwrap();

    let step = CoordinateTransform {
        meters_per_pixel: 0.01,
        invert_y: false,
        y_height: 0.0,
        x_columns: vec!["x".into()],
        y_columns: vec!["y".into()],
    };
    assert_eq!(step.name(), "coordinate_transform");

    let result = step.apply(df.lazy()).collect().unwrap();
    let x = result.column("x").unwrap().f64().unwrap();
    assert!((x.get(0).unwrap() - 1.0).abs() < 1e-10);
    assert!((x.get(1).unwrap() - 2.0).abs() < 1e-10);
}

#[test]
fn coordinate_transform_invert_y() {
    let df = df! {
        "x" => &[100.0],
        "y" => &[0.0],
    }
    .unwrap();

    let step = CoordinateTransform {
        meters_per_pixel: 0.01,
        invert_y: true,
        y_height: 200.0, // 200px
        x_columns: vec!["x".into()],
        y_columns: vec!["y".into()],
    };

    let result = step.apply(df.lazy()).collect().unwrap();
    let y = result.column("y").unwrap().f64().unwrap();
    // height_m = 200 * 0.01 = 2.0, pixel_to_meters(0) = 0.0, result = 2.0 - 0.0 = 2.0
    assert!((y.get(0).unwrap() - 2.0).abs() < 1e-10);
}

#[test]
fn midpoint_computer_averages() {
    let df = df! {
        "left_ear_x" => &[0.0, 2.0],
        "left_ear_y" => &[0.0, 4.0],
        "right_ear_x" => &[2.0, 6.0],
        "right_ear_y" => &[4.0, 8.0],
    }
    .unwrap();

    let step = MidpointComputer {
        groups: vec![MidpointGroup {
            output_label: "center_ear".into(),
            source_labels: vec!["left_ear".into(), "right_ear".into()],
        }],
    };
    assert_eq!(step.name(), "midpoint_computer");

    let result = step.apply(df.lazy()).collect().unwrap();
    let cx = result.column("center_ear_x").unwrap().f64().unwrap();
    let cy = result.column("center_ear_y").unwrap().f64().unwrap();
    assert!((cx.get(0).unwrap() - 1.0).abs() < 1e-10);
    assert!((cy.get(0).unwrap() - 2.0).abs() < 1e-10);
    assert!((cx.get(1).unwrap() - 4.0).abs() < 1e-10);
    assert!((cy.get(1).unwrap() - 6.0).abs() < 1e-10);
}

#[test]
fn median_smoother_smooths_columns() {
    let df = df! {
        "x" => &[1.0, 100.0, 1.0, 1.0, 1.0],
    }
    .unwrap();

    let step = MedianSmoother {
        window_size: 3,
        columns: vec!["x".into()],
    };
    assert_eq!(step.name(), "median_smoother");

    let result = step.apply(df.lazy()).collect().unwrap();
    let x = result.column("x").unwrap().f64().unwrap();
    // The spike at index 1 should be smoothed
    assert!((x.get(1).unwrap() - 1.0).abs() < 1e-10);
}
