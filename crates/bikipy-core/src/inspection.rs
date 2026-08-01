use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::video::VideoMetadata;

/// Geometry specification for a perimeter, serializable to JSON for Python inspection.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "shape")]
pub enum PerimeterSpec {
    #[serde(rename = "circle")]
    Circle {
        label: String,
        center_x: f64,
        center_y: f64,
        radius: f64,
    },
    #[serde(rename = "rectangle")]
    Rectangle {
        label: String,
        center_x: f64,
        center_y: f64,
        width: f64,
        height: f64,
    },
    #[serde(rename = "polygon")]
    Polygon {
        label: String,
        vertices: Vec<(f64, f64)>,
    },
    #[serde(rename = "radial_maze")]
    RadialMaze {
        label: String,
        center_vertices: Vec<(f64, f64)>,
        arms: Vec<Vec<(f64, f64)>>,
    },
    #[serde(rename = "triangle")]
    Triangle {
        label: String,
        vertices: [(f64, f64); 3],
    },
}

/// Metadata about a heuristic result column in the evaluation Parquet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeuristicMeta {
    pub name: String,
    pub result_column: String,
    pub true_frames: u64,
    pub total_frames: u64,
    pub seconds: f64,
}

/// Column mapping for a tracked label's coordinates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinateColumnSpec {
    pub x: String,
    pub y: String,
}

/// Full inspection manifest — everything the Python inspection package needs
/// to recreate plots from the Rust analysis output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InspectionManifest {
    pub video: VideoMetadata,
    pub video_path: Option<String>,
    pub labels: Vec<String>,
    pub label_colors: HashMap<String, String>,
    pub coordinate_columns: HashMap<String, CoordinateColumnSpec>,
    pub perimeters: Vec<PerimeterSpec>,
    pub heuristics: Vec<HeuristicMeta>,
    pub settings: InspectionSettings,
}

/// Runtime settings relevant to inspection/visualization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InspectionSettings {
    pub minimum_seconds_tolerance: f64,
    pub maximum_seconds_distraction: f64,
    pub meters_per_pixel: f64,
}

impl InspectionManifest {
    /// Write the manifest as JSON to the given path.
    pub fn write_json(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| crate::error::BikipyError::Other(e.to_string()))?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Read a manifest from a JSON file.
    pub fn from_json(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        serde_json::from_str(&content).map_err(|e| crate::error::BikipyError::Config(e.to_string()))
    }
}
