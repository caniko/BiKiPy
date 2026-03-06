use serde::{Deserialize, Serialize};

use crate::polygon::Polygon;
use bikipy_core::shape::{RayIntersectable, Shape};
use bikipy_core::types::BoundingBox;

/// A radial arm maze composed of multiple polygon arms branching from a center.
/// Used for Y-maze and multi-arm maze experiments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RadialMaze {
    pub arms: Vec<Polygon>,
    pub center: Polygon,
}

impl RadialMaze {
    pub fn new(center: Polygon, arms: Vec<Polygon>) -> Self {
        Self { arms, center }
    }

    /// Check which arm (if any) a point is in. Returns arm index.
    pub fn arm_containing(&self, x: f64, y: f64) -> Option<usize> {
        self.arms.iter().position(|arm| arm.contains(x, y))
    }

    /// True if the point is in the center region.
    pub fn in_center(&self, x: f64, y: f64) -> bool {
        self.center.contains(x, y)
    }
}

impl Shape for RadialMaze {
    fn contains(&self, x: f64, y: f64) -> bool {
        self.center.contains(x, y) || self.arms.iter().any(|arm| arm.contains(x, y))
    }

    fn bounding_box(&self) -> BoundingBox {
        let mut bb = self.center.bounding_box();
        for arm in &self.arms {
            let arm_bb = arm.bounding_box();
            bb.min_x = bb.min_x.min(arm_bb.min_x);
            bb.min_y = bb.min_y.min(arm_bb.min_y);
            bb.max_x = bb.max_x.max(arm_bb.max_x);
            bb.max_y = bb.max_y.max(arm_bb.max_y);
        }
        bb
    }
}

impl RayIntersectable for RadialMaze {
    fn ray_intersects(&self, origin: (f64, f64), direction: (f64, f64)) -> bool {
        self.center.ray_intersects(origin, direction)
            || self
                .arms
                .iter()
                .any(|arm| arm.ray_intersects(origin, direction))
    }
}
