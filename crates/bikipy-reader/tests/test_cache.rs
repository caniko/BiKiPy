use std::path::Path;

use bikipy_reader::cache::AugmentedCache;

#[test]
fn cache_path_generation() {
    let cache = AugmentedCache::new("/tmp/bikipy_cache");
    let path = cache.cache_path(Path::new("/data/experiment/trial1.csv"), "v1");
    assert_eq!(
        path.to_str().unwrap(),
        "/tmp/bikipy_cache/trial1_v1_augmented.parquet"
    );
}

#[test]
fn cache_path_with_different_suffix() {
    let cache = AugmentedCache::new("/tmp/cache");
    let path = cache.cache_path(Path::new("data.parquet"), "aug");
    assert!(path.to_str().unwrap().contains("data_aug_augmented.parquet"));
}

#[test]
fn cache_load_nonexistent_returns_none() {
    let cache = AugmentedCache::new("/tmp/bikipy_nonexistent_cache_dir");
    let result = cache
        .load(Path::new("/nonexistent/file.csv"), "test")
        .unwrap();
    assert!(result.is_none());
}

#[test]
fn cache_save_and_load_roundtrip() {
    use polars::prelude::*;

    let cache_dir = std::env::temp_dir().join("bikipy_cache_test");
    let cache = AugmentedCache::new(&cache_dir);

    let source = Path::new("trial42.csv");
    let mut df = df! { "x" => &[1.0, 2.0, 3.0] }.unwrap();

    cache.save(source, "v1", &mut df).unwrap();

    let loaded = cache.load(source, "v1").unwrap();
    assert!(loaded.is_some());
    let result = loaded.unwrap().collect().unwrap();
    assert_eq!(result.height(), 3);

    // Cleanup
    std::fs::remove_dir_all(&cache_dir).ok();
}
