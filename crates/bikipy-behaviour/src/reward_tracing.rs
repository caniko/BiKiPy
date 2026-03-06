use bikipy_feature::heuristic::Heuristic;
use bikipy_perimeter::circle::Circle;
use bikipy_perimeter::perimeter::Perimeter;

use crate::BehaviouralTask;

/// Cheeseboard / reward tracing task.
///
/// Animals search for baited wells on a board with multiple locations.
pub struct Cheeseboard {
    pub enclosure: Perimeter<Circle>,
    pub reward_locations: Vec<Perimeter<Circle>>,
}

impl BehaviouralTask for Cheeseboard {
    fn name(&self) -> &str {
        "cheeseboard"
    }

    fn heuristics(&self) -> Vec<Box<dyn Heuristic>> {
        // TODO: Proximity to reward wells, search pattern analysis
        Vec::new()
    }
}
