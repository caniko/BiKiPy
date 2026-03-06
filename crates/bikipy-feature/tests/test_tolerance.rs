use polars::prelude::*;

use bikipy_feature::tolerance::{apply_tolerance, bool_to_seconds};

#[test]
fn apply_tolerance_passthrough() {
    let df = df! {
        "active" => &[true, false, true, true, false],
    }
    .unwrap();

    let result = apply_tolerance(df.lazy(), "active", 2, 2, "tolerant")
        .collect()
        .unwrap();

    // Current implementation is a passthrough
    assert!(result.column("tolerant").is_ok());
    let t = result.column("tolerant").unwrap().bool().unwrap();
    assert_eq!(t.len(), 5);
}

#[test]
fn bool_to_seconds_counts() {
    let df = df! {
        "active" => &[true, true, true, false, false],
    }
    .unwrap();

    let seconds = bool_to_seconds(&df, "active", 30.0);
    assert!((seconds - 0.1).abs() < 1e-10); // 3 frames / 30 fps = 0.1s
}

#[test]
fn bool_to_seconds_all_true() {
    let df = df! {
        "active" => &[true, true, true, true],
    }
    .unwrap();

    let seconds = bool_to_seconds(&df, "active", 10.0);
    assert!((seconds - 0.4).abs() < 1e-10);
}

#[test]
fn bool_to_seconds_all_false() {
    let df = df! {
        "active" => &[false, false, false],
    }
    .unwrap();

    let seconds = bool_to_seconds(&df, "active", 30.0);
    assert!((seconds - 0.0).abs() < 1e-10);
}

#[test]
fn bool_to_seconds_missing_column() {
    let df = df! {
        "other" => &[1.0, 2.0],
    }
    .unwrap();

    let seconds = bool_to_seconds(&df, "nonexistent", 30.0);
    assert!((seconds - 0.0).abs() < 1e-10);
}
