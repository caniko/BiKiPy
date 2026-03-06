use std::path::Path;

use bikipy_reader::io::{scan_file, write_parquet, InputFormat};
use polars::prelude::*;

#[test]
fn input_format_from_csv() {
    let f = InputFormat::from_path(Path::new("data.csv")).unwrap();
    assert!(matches!(f, InputFormat::Csv));
}

#[test]
fn input_format_from_parquet() {
    let f = InputFormat::from_path(Path::new("data.parquet")).unwrap();
    assert!(matches!(f, InputFormat::Parquet));
}

#[test]
fn input_format_from_pq() {
    let f = InputFormat::from_path(Path::new("data.pq")).unwrap();
    assert!(matches!(f, InputFormat::Parquet));
}

#[test]
fn input_format_from_h5() {
    let f = InputFormat::from_path(Path::new("data.h5")).unwrap();
    assert!(matches!(f, InputFormat::Hdf5));
}

#[test]
fn input_format_unsupported() {
    let f = InputFormat::from_path(Path::new("data.xyz"));
    assert!(f.is_err());
}

#[test]
fn write_and_scan_parquet_roundtrip() {
    let mut df = df! {
        "a" => &[1.0, 2.0, 3.0],
        "b" => &["x", "y", "z"],
    }
    .unwrap();

    let tmp = std::env::temp_dir().join("bikipy_test_io_roundtrip.parquet");
    write_parquet(&mut df, &tmp).unwrap();

    let lf = scan_file(&tmp).unwrap();
    let result = lf.collect().unwrap();
    assert_eq!(result.height(), 3);
    assert_eq!(result.width(), 2);

    let a = result.column("a").unwrap().f64().unwrap();
    assert!((a.get(0).unwrap() - 1.0).abs() < 1e-10);

    std::fs::remove_file(&tmp).ok();
}

#[test]
fn scan_csv_file() {
    let tmp = std::env::temp_dir().join("bikipy_test_scan.csv");
    std::fs::write(&tmp, "x,y\n1.0,2.0\n3.0,4.0\n").unwrap();

    let lf = scan_file(&tmp).unwrap();
    let result = lf.collect().unwrap();
    assert_eq!(result.height(), 2);

    std::fs::remove_file(&tmp).ok();
}

#[test]
fn scan_hdf5_returns_error() {
    let tmp = std::env::temp_dir().join("bikipy_test.h5");
    std::fs::write(&tmp, "fake").unwrap();
    let result = scan_file(&tmp);
    assert!(result.is_err());
    std::fs::remove_file(&tmp).ok();
}
