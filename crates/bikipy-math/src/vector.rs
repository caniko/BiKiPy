use polars::prelude::*;

/// Compute the unit (normalized) vector from (x, y) columns.
/// Returns two new columns: `{prefix}_ux`, `{prefix}_uy`.
pub fn unit_vector_exprs(x_col: &str, y_col: &str, prefix: &str) -> Vec<Expr> {
    let magnitude = (col(x_col).pow(2) + col(y_col).pow(2)).sqrt();
    vec![
        (col(x_col) / magnitude.clone()).alias(format!("{prefix}_ux")),
        (col(y_col) / magnitude).alias(format!("{prefix}_uy")),
    ]
}

/// Compute the orthogonal (perpendicular) vector: (-y, x).
pub fn orthogonal_vector_exprs(x_col: &str, y_col: &str, prefix: &str) -> Vec<Expr> {
    vec![
        (lit(0.0) - col(y_col)).alias(format!("{prefix}_ortho_x")),
        col(x_col).alias(format!("{prefix}_ortho_y")),
    ]
}

/// Dot product of two 2D vector column pairs.
pub fn dot_product_expr(x1: &str, y1: &str, x2: &str, y2: &str) -> Expr {
    col(x1) * col(x2) + col(y1) * col(y2)
}

/// Direction vector from point A to point B columns.
pub fn direction_exprs(
    from_x: &str,
    from_y: &str,
    to_x: &str,
    to_y: &str,
    prefix: &str,
) -> Vec<Expr> {
    vec![
        (col(to_x) - col(from_x)).alias(format!("{prefix}_dx")),
        (col(to_y) - col(from_y)).alias(format!("{prefix}_dy")),
    ]
}

/// Rotate 2D vectors by an angle (in radians) using rotation matrix.
/// Returns rotated (x, y) as two new columns.
pub fn rotate_exprs(x_col: &str, y_col: &str, angle_rad: f64, prefix: &str) -> Vec<Expr> {
    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();
    vec![
        (col(x_col) * lit(cos_a) - col(y_col) * lit(sin_a)).alias(format!("{prefix}_rot_x")),
        (col(x_col) * lit(sin_a) + col(y_col) * lit(cos_a)).alias(format!("{prefix}_rot_y")),
    ]
}

/// Test whether a ray from `origin` in `direction` intersects a line segment
/// from `seg_start` to `seg_end`. Uses the parametric intersection method.
///
/// This is a scalar function used inside `Shape::ray_intersects` implementations.
pub fn ray_line_segment_intersection(
    origin: (f64, f64),
    direction: (f64, f64),
    seg_start: (f64, f64),
    seg_end: (f64, f64),
) -> bool {
    let (ox, oy) = origin;
    let (dx, dy) = direction;
    let (sx, sy) = seg_start;
    let (ex, ey) = seg_end;

    let seg_dx = ex - sx;
    let seg_dy = ey - sy;

    let denom = dx * seg_dy - dy * seg_dx;
    if denom.abs() < f64::EPSILON {
        return false; // Parallel
    }

    let t = ((sx - ox) * seg_dy - (sy - oy) * seg_dx) / denom;
    let u = ((sx - ox) * dy - (sy - oy) * dx) / denom;

    t >= 0.0 && (0.0..=1.0).contains(&u)
}

/// Compute the angle between two 2D vectors in radians.
pub fn angle_between(v1: (f64, f64), v2: (f64, f64)) -> f64 {
    let dot = v1.0 * v2.0 + v1.1 * v2.1;
    let mag1 = (v1.0 * v1.0 + v1.1 * v1.1).sqrt();
    let mag2 = (v2.0 * v2.0 + v2.1 * v2.1).sqrt();
    let cos_angle = (dot / (mag1 * mag2)).clamp(-1.0, 1.0);
    cos_angle.acos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ray_hits_segment() {
        assert!(ray_line_segment_intersection(
            (0.0, 0.0),
            (1.0, 0.0),
            (5.0, -1.0),
            (5.0, 1.0),
        ));
    }

    #[test]
    fn test_ray_misses_segment() {
        assert!(!ray_line_segment_intersection(
            (0.0, 0.0),
            (0.0, 1.0),
            (5.0, -1.0),
            (5.0, 1.0),
        ));
    }

    #[test]
    fn test_angle_between_orthogonal() {
        let angle = angle_between((1.0, 0.0), (0.0, 1.0));
        assert!((angle - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
    }
}
