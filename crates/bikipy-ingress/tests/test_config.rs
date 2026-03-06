use bikipy_ingress::config::{ExperimentConfig, IngressConfig};

#[test]
fn ingress_config_from_toml_string() {
    let toml_str = r#"
        [project]
        project_name = "test_project"
        data_directory = "/data"

        [[experiments]]
        name = "trial1"
        data_files = ["/data/trial1.csv"]
        task_type = "object_recognition"
        fps = 30.0
        meters_per_pixel = 0.001

        [[experiments]]
        name = "trial2"
        data_files = ["/data/trial2.csv", "/data/trial2b.csv"]
        task_type = "y_maze"
        fps = 25.0
        meters_per_pixel = 0.002
    "#;

    let config: IngressConfig = toml::from_str(toml_str).unwrap();
    assert_eq!(config.project.project_name, "test_project");
    assert_eq!(config.experiments.len(), 2);
    assert_eq!(config.experiments[0].name, "trial1");
    assert_eq!(config.experiments[0].task_type, "object_recognition");
    assert!((config.experiments[0].fps - 30.0).abs() < f64::EPSILON);
    assert_eq!(config.experiments[1].data_files.len(), 2);
}

#[test]
fn experiment_config_serde_roundtrip() {
    let exp = ExperimentConfig {
        name: "test".into(),
        data_files: vec!["/a.csv".into()],
        task_type: "cheeseboard".into(),
        fps: 30.0,
        meters_per_pixel: 0.001,
    };
    let json = serde_json::to_string(&exp).unwrap();
    let deser: ExperimentConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(deser.name, "test");
    assert_eq!(deser.task_type, "cheeseboard");
}

#[test]
fn ingress_config_from_file_nonexistent() {
    let result = IngressConfig::from_file(std::path::Path::new("/nonexistent/config.toml"));
    assert!(result.is_err());
}

#[test]
fn ingress_config_from_file_roundtrip() {
    let toml_str = r#"
        [project]
        project_name = "file_test"
        data_directory = "/data"

        [[experiments]]
        name = "exp1"
        data_files = ["/data/exp1.csv"]
        task_type = "object_recognition"
        fps = 30.0
        meters_per_pixel = 0.001
    "#;

    let tmp = std::env::temp_dir().join("bikipy_test_ingress_config.toml");
    std::fs::write(&tmp, toml_str).unwrap();

    let config = IngressConfig::from_file(&tmp).unwrap();
    assert_eq!(config.project.project_name, "file_test");
    assert_eq!(config.experiments.len(), 1);

    std::fs::remove_file(&tmp).ok();
}
