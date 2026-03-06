use serde::{Deserialize, Serialize};

use crate::polygon::Polygon;
use bikipy_core::shape::{Expandable, RayIntersectable, Shape};
use bikipy_core::types::BoundingBox;

/// A triangle defined by three vertices. Delegates to `Polygon` — DRY.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Triangle {
    polygon: Polygon,
}

impl Triangle {
    pub fn new(v1: (f64, f64), v2: (f64, f64), v3: (f64, f64)) -> Self {
        Self {
            polygon: Polygon::new(vec![v1, v2, v3]),
        }
    }

    pub fn vertices(&self) -> &[(f64, f64)] {
        &self.polygon.vertices
    }
}

impl Shape for Triangle {
    fn contains(&self, x: f64, y: f64) -> bool {
        self.polygon.contains(x, y)
    }

    fn bounding_box(&self) -> BoundingBox {
        self.polygon.bounding_box()
    }
}

impl Expandable for Triangle {
    type Output = Polygon;

    fn expand(&self, distance: f64) -> Polygon {
        self.polygon.expand(distance)
    }
}

impl RayIntersectable for Triangle {
    fn ray_intersects(&self, origin: (f64, f64), direction: (f64, f64)) -> bool {
        self.polygon.ray_intersects(origin, direction)
    }
}
