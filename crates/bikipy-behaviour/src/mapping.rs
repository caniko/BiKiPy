/// Map experiment names to behavioural task types.
///
/// Mirrors the Python `behaviour.mapping` module.
pub fn task_names() -> &'static [&'static str] {
    &[
        "object_recognition",
        "novel_object_recognition",
        "objects_in_updating_locations",
        "y_maze",
        "radial_arm",
        "cheeseboard",
        "reward_tracing",
    ]
}
