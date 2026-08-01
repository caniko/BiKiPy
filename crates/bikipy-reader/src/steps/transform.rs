use polars::prelude::*;

use crate::pipeline::PipelineStep;
use bikipy_math::geometry::pixel_to_meters_expr;

/// Convert pixel coordinates to meters and optionally invert the Y axis.
pub struct CoordinateTransform {
    pub meters_per_pixel: f64,
    pub invert_y: bool,
    pub y_height: f64,
    pub x_columns: Vec<String>,
    pub y_columns: Vec<String>,
}

impl PipelineStep for CoordinateTransform {
    fn name(&self) -> &str {
        "coordinate_transform"
    }

    fn apply(&self, lf: LazyFrame) -> LazyFrame {
        let mut lf = lf;

        // Convert all coordinate columns to meters
        for x_col in &self.x_columns {
            lf = lf.with_column(pixel_to_meters_expr(x_col, self.meters_per_pixel).alias(x_col));
        }
        for y_col in &self.y_columns {
            if self.invert_y {
                let height_m = self.y_height * self.meters_per_pixel;
                lf = lf.with_column(
                    (lit(height_m) - pixel_to_meters_expr(y_col, self.meters_per_pixel))
                        .alias(y_col),
                );
            } else {
                lf =
                    lf.with_column(pixel_to_meters_expr(y_col, self.meters_per_pixel).alias(y_col));
            }
        }

        lf
    }
}
