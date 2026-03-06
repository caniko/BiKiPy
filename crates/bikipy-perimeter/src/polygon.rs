use serde::{Deserialize, Serialize};

use bikipy_core::shape::{Expandable, RayIntersectable, Shape};
use bikipy_core::types::BoundingBox;
use bikipy_math::confinement::point_in_polygon;
use bikipy_math::vector::ray_line_segment_intersection;

/// A general polygon defined by ordered vertices.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Polygon {
    pub vertices: Vec<(f64, f64)>,
}

impl Polygon {
    pub fn new(vertices: Vec<(f64, f64)>) -> Self {
        Self { vertices }
    }

    /// Compute the centroid of this polygon.
    pub fn centroid(&self) -> (f64, f64) {
        let n = self.vertices.len() as f64;
        let (sx, sy) = self
            .vertices
            .iter()
            .fold((0.0, 0.0), |(ax, ay), &(x, y)| (ax + x, ay + y));
        (sx / n, sy / n)
    }

    /// Get edges as pairs of (start, end) vertices.
    pub fn edges(&self) -> impl Iterator<Item = ((f64, f64), (f64, f64))> + '_ {
        let n = self.vertices.len();
        (0..n).map(move |i| (self.vertices[i], self.vertices[(i + 1) % n]))
    }
}

impl Shape for Polygon {
    fn contains(&self, x: f64, y: f64) -> bool {
        point_in_polygon(x, y, &self.vertices)
    }

    fn bounding_box(&self) -> BoundingBox {
        let mut bb = BoundingBox {
            min_x: f64::INFINITY,
            min_y: f64::INFINITY,
            max_x: f64::NEG_INFINITY,
            max_y: f64::NEG_INFINITY,
        };
        for &(x, y) in &self.vertices {
            bb.min_x = bb.min_x.min(x);
            bb.min_y = bb.min_y.min(y);
            bb.max_x = bb.max_x.max(x);
            bb.max_y = bb.max_y.max(y);
        }
        bb
    }
}

impl Expandable for Polygon {
    type Output = Polygon;

    fn expand(&self, distance: f64) -> Polygon {
        // Expand by moving each vertex outward from centroid
        let (cx, cy) = self.centroid();
        let vertices = self
            .vertices
            .iter()
            .map(|&(x, y)| {
                let dx = x - cx;
                let dy = y - cy;
                let mag = (dx * dx + dy * dy).sqrt();
                if mag < f64::EPSILON {
                    (x, y)
                } else {
                    (x + distance * dx / mag, y + distance * dy / mag)
                }
            })
            .collect();
        Polygon { vertices }
    }
}

impl RayIntersectable for Polygon {
    fn ray_intersects(&self, origin: (f64, f64), direction: (f64, f64)) -> bool {
        self.edges()
            .any(|(start, end)| ray_line_segment_intersection(origin, direction, start, end))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_square() -> Polygon {
        Polygon::new(vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    }

    #[test]
    fn test_polygon_contains() {
        let sq = unit_square();
        assert!(sq.contains(0.5, 0.5));
        assert!(!sq.contains(1.5, 0.5));
    }

    #[test]
    fn test_polygon_centroid() {
        let sq = unit_square();
        let (cx, cy) = sq.centroid();
        assert!((cx - 0.5).abs() < f64::EPSILON);
        assert!((cy - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_polygon_expand() {
        let sq = unit_square();
        let expanded = sq.expand(0.1);
        // Expanded polygon should contain original vertices
        assert!(expanded.contains(0.5, 0.5));
        // And points slightly outside original
        assert!(expanded.contains(-0.05, 0.5));
    }

    #[test]
    fn test_polygon_ray_intersection() {
        let sq = unit_square();
        assert!(sq.ray_intersects((-1.0, 0.5), (1.0, 0.0)));
        assert!(!sq.ray_intersects((-1.0, 2.0), (1.0, 0.0)));
    }
}
