use bikipy_core::config::*;

#[test]
fn runtime_settings_defaults() {
    let settings = RuntimeSettings::default();
    assert!((settings.minimum_seconds_tolerance - 0.5).abs() < f64::EPSILON);
    assert!((settings.maximum_seconds_distraction - 1.0 / 3.0).abs() < f64::EPSILON);
    assert!(settings.num_threads.is_none());
}

#[test]
fn runtime_settings_serde_with_defaults() {
    let toml_str = "";
    let settings: RuntimeSettings = toml::from_str(toml_str).unwrap();
    assert!((settings.minimum_seconds_tolerance - 0.5).abs() < f64::EPSILON);
    assert!(settings.num_threads.is_none());
}

#[test]
fn runtime_settings_serde_with_overrides() {
    let toml_str = r#"
        minimum_seconds_tolerance = 1.0
        maximum_seconds_distraction = 0.5
        num_threads = 4
    "#;
    let settings: RuntimeSettings = toml::from_str(toml_str).unwrap();
    assert!((settings.minimum_seconds_tolerance - 1.0).abs() < f64::EPSILON);
    assert!((settings.maximum_seconds_distraction - 0.5).abs() < f64::EPSILON);
    assert_eq!(settings.num_threads, Some(4));
}

#[test]
fn project_config_serde() {
    let toml_str = r#"
        project_name = "my_experiment"
        data_directory = "/data/experiment1"
    "#;
    let config: ProjectConfig = toml::from_str(toml_str).unwrap();
    assert_eq!(config.project_name, "my_experiment");
    assert_eq!(config.data_directory, "/data/experiment1");
    // Runtime should be defaults
    assert!((config.runtime.minimum_seconds_tolerance - 0.5).abs() < f64::EPSILON);
}

#[test]
fn project_config_with_runtime() {
    let toml_str = r#"
        project_name = "test"
        data_directory = "/tmp"
        [runtime]
        minimum_seconds_tolerance = 2.0
        num_threads = 8
    "#;
    let config: ProjectConfig = toml::from_str(toml_str).unwrap();
    assert_eq!(config.project_name, "test");
    assert!((config.runtime.minimum_seconds_tolerance - 2.0).abs() < f64::EPSILON);
    assert_eq!(config.runtime.num_threads, Some(8));
}

#[test]
fn project_config_roundtrip() {
    let config = ProjectConfig {
        project_name: "roundtrip".to_string(),
        data_directory: "/data".to_string(),
        runtime: RuntimeSettings::default(),
    };
    let serialized = toml::to_string(&config).unwrap();
    let deserialized: ProjectConfig = toml::from_str(&serialized).unwrap();
    assert_eq!(deserialized.project_name, "roundtrip");
}
