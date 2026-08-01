use std::collections::HashMap;

use bikipy_core::inspection::*;
use bikipy_core::types::MetersPerPixel;
use bikipy_core::video::VideoMetadata;

fn sample_manifest() -> InspectionManifest {
    let mut coord_cols = HashMap::new();
    coord_cols.insert(
        "nose".to_string(),
        CoordinateColumnSpec {
            x: "nose_x".to_string(),
            y: "nose_y".to_string(),
        },
    );

    let mut label_colors = HashMap::new();
    label_colors.insert("nose".to_string(), "#1f77b4".to_string());

    InspectionManifest {
        video: VideoMetadata {
            fps: 30.0,
            total_frames: 9000,
            resolution: (1920, 1080),
            meters_per_pixel: MetersPerPixel(0.001),
        },
        video_path: Some("/data/video.mp4".to_string()),
        labels: vec!["nose".to_string(), "tail_base".to_string()],
        label_colors,
        coordinate_columns: coord_cols,
        perimeters: vec![
            PerimeterSpec::Circle {
                label: "arena".to_string(),
                center_x: 0.15,
                center_y: 0.12,
                radius: 0.14,
            },
            PerimeterSpec::Rectangle {
                label: "zone_a".to_string(),
                center_x: 0.1,
                center_y: 0.1,
                width: 0.05,
                height: 0.04,
            },
        ],
        heuristics: vec![HeuristicMeta {
            name: "body_proximity".to_string(),
            result_column: "body_proximity".to_string(),
            true_frames: 450,
            total_frames: 9000,
            seconds: 15.0,
        }],
        settings: InspectionSettings {
            minimum_seconds_tolerance: 0.5,
            maximum_seconds_distraction: 1.0 / 3.0,
            meters_per_pixel: 0.001,
        },
    }
}

#[test]
fn manifest_serde_json_roundtrip() {
    let manifest = sample_manifest();
    let json = serde_json::to_string_pretty(&manifest).unwrap();
    let deser: InspectionManifest = serde_json::from_str(&json).unwrap();

    assert_eq!(deser.labels, manifest.labels);
    assert_eq!(deser.heuristics.len(), 1);
    assert_eq!(deser.heuristics[0].name, "body_proximity");
    assert_eq!(deser.heuristics[0].true_frames, 450);
    assert_eq!(deser.perimeters.len(), 2);
    assert!(deser.video_path.is_some());
    assert_eq!(deser.coordinate_columns.len(), 1);
    assert_eq!(deser.coordinate_columns["nose"].x, "nose_x");
}

#[test]
fn perimeter_spec_circle_tagged_serde() {
    let spec = PerimeterSpec::Circle {
        label: "arena".into(),
        center_x: 1.0,
        center_y: 2.0,
        radius: 3.0,
    };
    let json = serde_json::to_string(&spec).unwrap();
    assert!(json.contains(r#""shape":"circle"#));

    let deser: PerimeterSpec = serde_json::from_str(&json).unwrap();
    match deser {
        PerimeterSpec::Circle {
            label,
            center_x,
            radius,
            ..
        } => {
            assert_eq!(label, "arena");
            assert!((center_x - 1.0).abs() < f64::EPSILON);
            assert!((radius - 3.0).abs() < f64::EPSILON);
        }
        _ => panic!("expected Circle"),
    }
}

#[test]
fn perimeter_spec_rectangle_tagged_serde() {
    let spec = PerimeterSpec::Rectangle {
        label: "zone".into(),
        center_x: 0.0,
        center_y: 0.0,
        width: 1.0,
        height: 2.0,
    };
    let json = serde_json::to_string(&spec).unwrap();
    assert!(json.contains(r#""shape":"rectangle"#));
}

#[test]
fn perimeter_spec_polygon_tagged_serde() {
    let spec = PerimeterSpec::Polygon {
        label: "pen".into(),
        vertices: vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
    };
    let json = serde_json::to_string(&spec).unwrap();
    assert!(json.contains(r#""shape":"polygon"#));

    let deser: PerimeterSpec = serde_json::from_str(&json).unwrap();
    match deser {
        PerimeterSpec::Polygon { vertices, .. } => assert_eq!(vertices.len(), 3),
        _ => panic!("expected Polygon"),
    }
}

#[test]
fn perimeter_spec_triangle_tagged_serde() {
    let spec = PerimeterSpec::Triangle {
        label: "tri".into(),
        vertices: [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)],
    };
    let json = serde_json::to_string(&spec).unwrap();
    assert!(json.contains(r#""shape":"triangle"#));
}

#[test]
fn perimeter_spec_radial_maze_tagged_serde() {
    let spec = PerimeterSpec::RadialMaze {
        label: "ymaze".into(),
        center_vertices: vec![(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)],
        arms: vec![vec![(2.0, 0.0), (3.0, 0.0), (3.0, 1.0), (2.0, 1.0)]],
    };
    let json = serde_json::to_string(&spec).unwrap();
    assert!(json.contains(r#""shape":"radial_maze"#));

    let deser: PerimeterSpec = serde_json::from_str(&json).unwrap();
    match deser {
        PerimeterSpec::RadialMaze { arms, .. } => assert_eq!(arms.len(), 1),
        _ => panic!("expected RadialMaze"),
    }
}

#[test]
fn heuristic_meta_serde() {
    let meta = HeuristicMeta {
        name: "test".into(),
        result_column: "test_bool".into(),
        true_frames: 100,
        total_frames: 1000,
        seconds: 3.33,
    };
    let json = serde_json::to_string(&meta).unwrap();
    let deser: HeuristicMeta = serde_json::from_str(&json).unwrap();
    assert_eq!(deser.name, "test");
    assert_eq!(deser.result_column, "test_bool");
    assert_eq!(deser.true_frames, 100);
}

#[test]
fn manifest_write_and_read_json() {
    let manifest = sample_manifest();
    let tmp = std::env::temp_dir().join("bikipy_test_inspection_manifest.json");

    manifest.write_json(&tmp).unwrap();
    assert!(tmp.exists());

    let loaded = InspectionManifest::from_json(&tmp).unwrap();
    assert_eq!(loaded.labels, manifest.labels);
    assert_eq!(loaded.heuristics.len(), manifest.heuristics.len());
    assert_eq!(loaded.perimeters.len(), manifest.perimeters.len());
    assert_eq!(
        loaded.coordinate_columns["nose"].x,
        manifest.coordinate_columns["nose"].x,
    );

    std::fs::remove_file(&tmp).ok();
}

#[test]
fn manifest_from_json_nonexistent_file() {
    let result = InspectionManifest::from_json(std::path::Path::new("/nonexistent/file.json"));
    assert!(result.is_err());
}

#[test]
fn manifest_from_json_invalid_content() {
    let tmp = std::env::temp_dir().join("bikipy_test_bad_manifest.json");
    std::fs::write(&tmp, "not valid json").unwrap();

    let result = InspectionManifest::from_json(&tmp);
    assert!(result.is_err());

    std::fs::remove_file(&tmp).ok();
}

#[test]
fn inspection_settings_serde() {
    let settings = InspectionSettings {
        minimum_seconds_tolerance: 0.5,
        maximum_seconds_distraction: 0.333,
        meters_per_pixel: 0.001,
    };
    let json = serde_json::to_string(&settings).unwrap();
    let deser: InspectionSettings = serde_json::from_str(&json).unwrap();
    assert!((deser.minimum_seconds_tolerance - 0.5).abs() < f64::EPSILON);
    assert!((deser.meters_per_pixel - 0.001).abs() < f64::EPSILON);
}

#[test]
fn manifest_empty_collections() {
    let manifest = InspectionManifest {
        video: VideoMetadata {
            fps: 30.0,
            total_frames: 0,
            resolution: (0, 0),
            meters_per_pixel: MetersPerPixel(0.0),
        },
        video_path: None,
        labels: vec![],
        label_colors: HashMap::new(),
        coordinate_columns: HashMap::new(),
        perimeters: vec![],
        heuristics: vec![],
        settings: InspectionSettings {
            minimum_seconds_tolerance: 0.5,
            maximum_seconds_distraction: 0.333,
            meters_per_pixel: 0.0,
        },
    };
    let json = serde_json::to_string(&manifest).unwrap();
    let deser: InspectionManifest = serde_json::from_str(&json).unwrap();
    assert!(deser.labels.is_empty());
    assert!(deser.perimeters.is_empty());
    assert!(deser.video_path.is_none());
}
