use serde::{Deserialize, Serialize};

use bikipy_core::shape::{Expandable, RayIntersectable, Shape};
use bikipy_core::types::BoundingBox;

/// A circle defined by center and radius.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Circle {
    pub center_x: f64,
    pub center_y: f64,
    pub radius: f64,
}

impl Circle {
    pub fn new(center_x: f64, center_y: f64, radius: f64) -> Self {
        Self {
            center_x,
            center_y,
            radius,
        }
    }
}

impl Shape for Circle {
    fn contains(&self, x: f64, y: f64) -> bool {
        let dx = x - self.center_x;
        let dy = y - self.center_y;
        dx * dx + dy * dy <= self.radius * self.radius
    }

    fn bounding_box(&self) -> BoundingBox {
        BoundingBox {
            min_x: self.center_x - self.radius,
            min_y: self.center_y - self.radius,
            max_x: self.center_x + self.radius,
            max_y: self.center_y + self.radius,
        }
    }
}

impl Expandable for Circle {
    type Output = Circle;

    fn expand(&self, distance: f64) -> Circle {
        Circle {
            center_x: self.center_x,
            center_y: self.center_y,
            radius: self.radius + distance,
        }
    }
}

impl RayIntersectable for Circle {
    fn ray_intersects(&self, origin: (f64, f64), direction: (f64, f64)) -> bool {
        let (ox, oy) = origin;
        let (dx, dy) = direction;

        // Vector from ray origin to circle center
        let fx = ox - self.center_x;
        let fy = oy - self.center_y;

        let a = dx * dx + dy * dy;
        let b = 2.0 * (fx * dx + fy * dy);
        let c = fx * fx + fy * fy - self.radius * self.radius;

        let discriminant = b * b - 4.0 * a * c;
        if discriminant < 0.0 {
            return false;
        }

        let sqrt_disc = discriminant.sqrt();
        let t1 = (-b - sqrt_disc) / (2.0 * a);
        let t2 = (-b + sqrt_disc) / (2.0 * a);

        // Ray hits if either intersection is at t >= 0 (ahead of origin)
        t1 >= 0.0 || t2 >= 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circle_contains() {
        let c = Circle::new(0.0, 0.0, 1.0);
        assert!(c.contains(0.0, 0.0));
        assert!(c.contains(0.5, 0.5));
        assert!(!c.contains(1.5, 0.0));
    }

    #[test]
    fn test_circle_expand() {
        let c = Circle::new(0.0, 0.0, 1.0);
        let expanded = c.expand(0.5);
        assert!((expanded.radius - 1.5).abs() < f64::EPSILON);
        assert!(expanded.contains(1.3, 0.0));
    }

    #[test]
    fn test_circle_ray_hit() {
        let c = Circle::new(5.0, 0.0, 1.0);
        assert!(c.ray_intersects((0.0, 0.0), (1.0, 0.0)));
    }

    #[test]
    fn test_circle_ray_miss() {
        let c = Circle::new(5.0, 5.0, 1.0);
        assert!(!c.ray_intersects((0.0, 0.0), (1.0, 0.0)));
    }
}
