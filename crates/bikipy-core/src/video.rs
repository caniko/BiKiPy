use serde::{Deserialize, Serialize};

use crate::types::MetersPerPixel;

/// Metadata about a video recording session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoMetadata {
    /// Frames per second of the recording.
    pub fps: f64,

    /// Total number of frames.
    pub total_frames: u64,

    /// Video resolution (width, height) in pixels.
    pub resolution: (u32, u32),

    /// Conversion factor from pixels to meters.
    pub meters_per_pixel: MetersPerPixel,
}

impl VideoMetadata {
    /// Convert a frame count to seconds.
    pub fn frames_to_seconds(&self, frames: u64) -> f64 {
        frames as f64 / self.fps
    }

    /// Convert seconds to frame count.
    pub fn seconds_to_frames(&self, seconds: f64) -> u64 {
        (seconds * self.fps).round() as u64
    }
}
