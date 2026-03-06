/// Point-in-polygon test using ray-casting algorithm.
/// Vertices must be ordered (clockwise or counter-clockwise).
pub fn point_in_polygon(x: f64, y: f64, vertices: &[(f64, f64)]) -> bool {
    let n = vertices.len();
    if n < 3 {
        return false;
    }

    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = vertices[i];
        let (xj, yj) = vertices[j];

        if ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Point-in-ellipse test.
/// Returns true if (x, y) is inside the ellipse centered at (cx, cy)
/// with semi-axes (rx, ry).
pub fn point_in_ellipse(x: f64, y: f64, cx: f64, cy: f64, rx: f64, ry: f64) -> bool {
    let dx = (x - cx) / rx;
    let dy = (y - cy) / ry;
    dx * dx + dy * dy <= 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_in_square() {
        let square = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        assert!(point_in_polygon(0.5, 0.5, &square));
        assert!(!point_in_polygon(1.5, 0.5, &square));
    }

    #[test]
    fn test_point_in_ellipse_center() {
        assert!(point_in_ellipse(0.0, 0.0, 0.0, 0.0, 1.0, 1.0));
    }

    #[test]
    fn test_point_outside_ellipse() {
        assert!(!point_in_ellipse(2.0, 0.0, 0.0, 0.0, 1.0, 1.0));
    }
}
