use bikipy_feature::heuristic::Heuristic;
use bikipy_perimeter::perimeter::Perimeter;
use bikipy_perimeter::radial_maze::RadialMaze;

use crate::BehaviouralTask;

/// Y-maze / radial arm maze task.
///
/// Tracks which arms the animal visits and computes exploration patterns.
pub struct YMaze {
    pub maze: Perimeter<RadialMaze>,
}

impl BehaviouralTask for YMaze {
    fn name(&self) -> &str {
        "y_maze"
    }

    fn heuristics(&self) -> Vec<Box<dyn Heuristic>> {
        // TODO: Arm visit heuristics, alternation scoring
        Vec::new()
    }
}
