use bikipy_core::types::MetersPerPixel;
use bikipy_core::video::VideoMetadata;

fn sample_video() -> VideoMetadata {
    VideoMetadata {
        fps: 30.0,
        total_frames: 9000,
        resolution: (1920, 1080),
        meters_per_pixel: MetersPerPixel(0.001),
    }
}

#[test]
fn frames_to_seconds() {
    let v = sample_video();
    assert!((v.frames_to_seconds(30) - 1.0).abs() < f64::EPSILON);
    assert!((v.frames_to_seconds(0) - 0.0).abs() < f64::EPSILON);
    assert!((v.frames_to_seconds(90) - 3.0).abs() < f64::EPSILON);
}

#[test]
fn seconds_to_frames() {
    let v = sample_video();
    assert_eq!(v.seconds_to_frames(1.0), 30);
    assert_eq!(v.seconds_to_frames(0.0), 0);
    assert_eq!(v.seconds_to_frames(3.0), 90);
}

#[test]
fn frames_seconds_roundtrip() {
    let v = sample_video();
    let frames = 150u64;
    let seconds = v.frames_to_seconds(frames);
    let back = v.seconds_to_frames(seconds);
    assert_eq!(back, frames);
}

#[test]
fn video_metadata_serde_roundtrip() {
    let v = sample_video();
    let json = serde_json::to_string(&v).unwrap();
    let deserialized: VideoMetadata = serde_json::from_str(&json).unwrap();
    assert!((deserialized.fps - 30.0).abs() < f64::EPSILON);
    assert_eq!(deserialized.total_frames, 9000);
    assert_eq!(deserialized.resolution, (1920, 1080));
}

#[test]
fn total_duration() {
    let v = sample_video();
    let duration = v.frames_to_seconds(v.total_frames);
    assert!((duration - 300.0).abs() < f64::EPSILON); // 9000/30 = 300s
}
