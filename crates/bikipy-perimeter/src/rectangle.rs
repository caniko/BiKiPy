use serde::{Deserialize, Serialize};

use crate::polygon::Polygon;
use bikipy_core::shape::{Expandable, RayIntersectable, Shape};
use bikipy_core::types::BoundingBox;

/// A rectangle defined by center, width, height, and optional rotation.
/// Delegates to `Polygon` for all geometric operations — DRY.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Rectangle {
    pub center_x: f64,
    pub center_y: f64,
    pub width: f64,
    pub height: f64,
    polygon: Polygon,
}

impl Rectangle {
    pub fn new(center_x: f64, center_y: f64, width: f64, height: f64) -> Self {
        let hw = width / 2.0;
        let hh = height / 2.0;
        let polygon = Polygon::new(vec![
            (center_x - hw, center_y - hh),
            (center_x + hw, center_y - hh),
            (center_x + hw, center_y + hh),
            (center_x - hw, center_y + hh),
        ]);
        Self {
            center_x,
            center_y,
            width,
            height,
            polygon,
        }
    }
}

// All Shape/Expandable/RayIntersectable delegated to inner Polygon — DRY.
impl Shape for Rectangle {
    fn contains(&self, x: f64, y: f64) -> bool {
        self.polygon.contains(x, y)
    }

    fn bounding_box(&self) -> BoundingBox {
        self.polygon.bounding_box()
    }
}

impl Expandable for Rectangle {
    type Output = Polygon;

    fn expand(&self, distance: f64) -> Polygon {
        self.polygon.expand(distance)
    }
}

impl RayIntersectable for Rectangle {
    fn ray_intersects(&self, origin: (f64, f64), direction: (f64, f64)) -> bool {
        self.polygon.ray_intersects(origin, direction)
    }
}
