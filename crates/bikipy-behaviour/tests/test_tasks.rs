use bikipy_behaviour::BehaviouralTask;
use bikipy_behaviour::mapping::task_names;
use bikipy_behaviour::object_recognition::ObjectRecognition;
use bikipy_behaviour::radial_arm::YMaze;
use bikipy_behaviour::reward_tracing::Cheeseboard;
use bikipy_perimeter::circle::Circle;
use bikipy_perimeter::perimeter::Perimeter;
use bikipy_perimeter::polygon::Polygon;
use bikipy_perimeter::radial_maze::RadialMaze;

#[test]
fn task_names_contains_expected() {
    let names = task_names();
    assert!(names.contains(&"object_recognition"));
    assert!(names.contains(&"y_maze"));
    assert!(names.contains(&"cheeseboard"));
    assert!(names.len() >= 5);
}

#[test]
fn object_recognition_name() {
    let task = ObjectRecognition {
        object_perimeters: vec![],
        max_distance: 0.05,
        max_angle: 45.0,
    };
    assert_eq!(task.name(), "object_recognition");
}

#[test]
fn object_recognition_heuristics_empty_objects() {
    let task = ObjectRecognition {
        object_perimeters: vec![],
        max_distance: 0.05,
        max_angle: 45.0,
    };
    assert_eq!(task.heuristics().len(), 0);
}

#[test]
fn object_recognition_heuristics_with_objects() {
    let task = ObjectRecognition {
        object_perimeters: vec![
            Perimeter::new("obj1", Circle::new(1.0, 1.0, 0.5)),
            Perimeter::new("obj2", Circle::new(3.0, 3.0, 0.5)),
        ],
        max_distance: 0.05,
        max_angle: 45.0,
    };
    // 4 heuristics per object: body, whisker, olfaction, outside
    assert_eq!(task.heuristics().len(), 8);
}

#[test]
fn y_maze_name() {
    let center = Polygon::new(vec![(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]);
    let maze = RadialMaze::new(center, vec![]);
    let task = YMaze {
        maze: Perimeter::new("ymaze", maze),
    };
    assert_eq!(task.name(), "y_maze");
    assert_eq!(task.heuristics().len(), 0); // stub
}

#[test]
fn cheeseboard_name() {
    let task = Cheeseboard {
        enclosure: Perimeter::new("board", Circle::new(0.0, 0.0, 5.0)),
        reward_locations: vec![],
    };
    assert_eq!(task.name(), "cheeseboard");
    assert_eq!(task.heuristics().len(), 0); // stub
}

#[test]
fn analysis_result_serde() {
    let result = bikipy_behaviour::AnalysisResult {
        task_name: "test".into(),
        summaries: vec![],
        evaluated_df: None,
    };
    let json = serde_json::to_string(&result).unwrap();
    let deser: bikipy_behaviour::AnalysisResult = serde_json::from_str(&json).unwrap();
    assert_eq!(deser.task_name, "test");
    assert!(deser.evaluated_df.is_none()); // skipped in serde
}
