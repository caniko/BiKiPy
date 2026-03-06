use bikipy_feature::heuristic::Heuristic;
use bikipy_feature::heuristic::helper::outside_perimeter::OutsidePerimeterHeuristic;
use bikipy_feature::heuristic::solo::body_proximity::BodyProximityHeuristic;
use bikipy_feature::heuristic::solo::olfaction::OlfactionHeuristic;
use bikipy_feature::heuristic::solo::whiskers::WhiskerHeuristic;
use bikipy_core::types::CoordinateColumns;
use bikipy_perimeter::circle::Circle;
use bikipy_perimeter::perimeter::Perimeter;

use crate::BehaviouralTask;

/// Novel Object Recognition task.
///
/// Animals explore objects; heuristics detect investigation via
/// body proximity, whisker contact, and olfactory sniffing.
pub struct ObjectRecognition {
    pub object_perimeters: Vec<Perimeter<Circle>>,
    pub max_distance: f64,
    pub max_angle: f64,
}

impl BehaviouralTask for ObjectRecognition {
    fn name(&self) -> &str {
        "object_recognition"
    }

    fn heuristics(&self) -> Vec<Box<dyn Heuristic>> {
        let mut heuristics: Vec<Box<dyn Heuristic>> = Vec::new();

        for perimeter in &self.object_perimeters {
            // Each object gets all three heuristic types
            let body = BodyProximityHeuristic {
                perimeter: perimeter.clone(),
                max_distance: self.max_distance,
            };

            let whisker = WhiskerHeuristic {
                perimeter: perimeter.clone(),
                max_distance: self.max_distance,
                max_angle: self.max_angle,
            };

            let olfaction = OlfactionHeuristic {
                perimeter: perimeter.clone(),
                max_distance: self.max_distance,
                max_angle: self.max_angle,
            };

            let outside = OutsidePerimeterHeuristic {
                perimeter: perimeter.clone(),
                coords: CoordinateColumns::new("center_ear_x", "center_ear_y"),
            };

            // Combined: (proximity OR whisker OR olfaction) AND outside
            heuristics.push(Box::new(body));
            heuristics.push(Box::new(whisker));
            heuristics.push(Box::new(olfaction));
            heuristics.push(Box::new(outside));
        }

        heuristics
    }
}
