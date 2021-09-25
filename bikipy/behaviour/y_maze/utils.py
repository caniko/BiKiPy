import numpy as np


def mean_intersecting_points_on_borders(arms, center):
    for center_idx, corner in enumerate(center.corners):
        corner = np.asarray(corner)
        closest_sides, closest_idxs, closest_distances = [], [], []
        for arm in arms:
            vectors = np.array(arm.corners) - corner
            distance = np.linalg.norm(vectors, axis=1)

            # Find the closest value to the corner in the center perimeter
            closest_side_idx = np.where(np.argsort(distance) == 0)[0][0]
            closest_idxs.append(closest_side_idx)

            closest_distances.append(distance[closest_side_idx])
            closest_sides.append(arm.corners[closest_side_idx])

        # Only two arms intersect at corner, need to eliminate the 3rd non-intersecting one
        closest_sides = np.array(closest_sides)
        argsorted_distances = np.argsort(closest_distances)

        intersecting_sides = closest_sides[argsorted_distances][:2]
        average_position = np.mean((*intersecting_sides, corner), axis=0)

        # Find the index of the arm that is non-intersecting
        non_intersecting_arm = np.where(argsorted_distances == 2)[0][0]

        for arm_idx, closest_side_idx in enumerate(closest_idxs):
            if non_intersecting_arm == arm_idx:
                continue

            current_sides = list(arms[arm_idx].corners)
            current_sides[closest_side_idx] = average_position
            arms[arm_idx].corners = current_sides

        center_sides = list(center.corners)
        center_sides[center_idx] = average_position
        center.corners = center_sides

    return arms, center
