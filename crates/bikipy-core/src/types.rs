use serde::{Deserialize, Serialize};

/// A named label for a tracked body part (e.g. "nose", "left_ear", "tail_base").
pub type Label = String;

/// Conversion factor from pixels to meters.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MetersPerPixel(pub f64);

/// Axis-aligned bounding box for spatial queries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
}

impl BoundingBox {
    pub fn contains(&self, x: f64, y: f64) -> bool {
        x >= self.min_x && x <= self.max_x && y >= self.min_y && y <= self.max_y
    }

    pub fn expand(&self, distance: f64) -> Self {
        Self {
            min_x: self.min_x - distance,
            min_y: self.min_y - distance,
            max_x: self.max_x + distance,
            max_y: self.max_y + distance,
        }
    }
}

/// Specification for a coordinate column pair in a DataFrame.
#[derive(Debug, Clone)]
pub struct CoordinateColumns {
    pub x: String,
    pub y: String,
}

impl CoordinateColumns {
    pub fn new(x: impl Into<String>, y: impl Into<String>) -> Self {
        Self {
            x: x.into(),
            y: y.into(),
        }
    }
}

/// Specification for ray origin + direction columns.
#[derive(Debug, Clone)]
pub struct RaySpec {
    pub origin: CoordinateColumns,
    pub direction: CoordinateColumns,
}

/// Coordinate enum for multi-indexed dataframe access.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Coordinate {
    X,
    Y,
    Likelihood,
}

impl Coordinate {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::X => "x",
            Self::Y => "y",
            Self::Likelihood => "likelihood",
        }
    }
}

/// Group of body-part labels whose midpoint should be computed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MidpointGroup {
    pub output_label: Label,
    pub source_labels: Vec<Label>,
}
