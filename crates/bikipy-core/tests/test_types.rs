use bikipy_core::types::*;

#[test]
fn bounding_box_contains_inside() {
    let bb = BoundingBox {
        min_x: 0.0,
        min_y: 0.0,
        max_x: 10.0,
        max_y: 10.0,
    };
    assert!(bb.contains(5.0, 5.0));
    assert!(bb.contains(0.0, 0.0));
    assert!(bb.contains(10.0, 10.0));
}

#[test]
fn bounding_box_contains_outside() {
    let bb = BoundingBox {
        min_x: 0.0,
        min_y: 0.0,
        max_x: 10.0,
        max_y: 10.0,
    };
    assert!(!bb.contains(-1.0, 5.0));
    assert!(!bb.contains(5.0, 11.0));
    assert!(!bb.contains(11.0, 5.0));
    assert!(!bb.contains(5.0, -1.0));
}

#[test]
fn bounding_box_expand() {
    let bb = BoundingBox {
        min_x: 1.0,
        min_y: 2.0,
        max_x: 3.0,
        max_y: 4.0,
    };
    let expanded = bb.expand(0.5);
    assert!((expanded.min_x - 0.5).abs() < f64::EPSILON);
    assert!((expanded.min_y - 1.5).abs() < f64::EPSILON);
    assert!((expanded.max_x - 3.5).abs() < f64::EPSILON);
    assert!((expanded.max_y - 4.5).abs() < f64::EPSILON);
    // Expanded box should contain points just outside original
    assert!(expanded.contains(0.6, 1.6));
    assert!(!bb.contains(0.6, 1.6));
}

#[test]
fn bounding_box_expand_negative() {
    let bb = BoundingBox {
        min_x: 0.0,
        min_y: 0.0,
        max_x: 10.0,
        max_y: 10.0,
    };
    let contracted = bb.expand(-1.0);
    assert!((contracted.min_x - 1.0).abs() < f64::EPSILON);
    assert!((contracted.max_x - 9.0).abs() < f64::EPSILON);
    assert!(!contracted.contains(0.5, 5.0));
}

#[test]
fn coordinate_columns_new() {
    let cc = CoordinateColumns::new("nose_x", "nose_y");
    assert_eq!(cc.x, "nose_x");
    assert_eq!(cc.y, "nose_y");
}

#[test]
fn coordinate_columns_from_string() {
    let cc = CoordinateColumns::new(String::from("a"), String::from("b"));
    assert_eq!(cc.x, "a");
    assert_eq!(cc.y, "b");
}

#[test]
fn coordinate_enum_as_str() {
    assert_eq!(Coordinate::X.as_str(), "x");
    assert_eq!(Coordinate::Y.as_str(), "y");
    assert_eq!(Coordinate::Likelihood.as_str(), "likelihood");
}

#[test]
fn coordinate_enum_equality() {
    assert_eq!(Coordinate::X, Coordinate::X);
    assert_ne!(Coordinate::X, Coordinate::Y);
}

#[test]
fn meters_per_pixel_serde_roundtrip() {
    let mpp = MetersPerPixel(0.001);
    let json = serde_json::to_string(&mpp).unwrap();
    let deserialized: MetersPerPixel = serde_json::from_str(&json).unwrap();
    assert_eq!(mpp, deserialized);
}

#[test]
fn midpoint_group_serde() {
    let mg = MidpointGroup {
        output_label: "center_ear".to_string(),
        source_labels: vec!["left_ear".to_string(), "right_ear".to_string()],
    };
    let json = serde_json::to_string(&mg).unwrap();
    let deserialized: MidpointGroup = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.output_label, "center_ear");
    assert_eq!(deserialized.source_labels.len(), 2);
}
