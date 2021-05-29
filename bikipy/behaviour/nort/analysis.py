#     or_filter = location_filtered | gaze_filtered
#     frames = or_filter.size
#
#     ana_loc = location_filtered[or_filter]
#     ana_gaze = gaze_filtered[or_filter]
#     ana_obs = object_observation[or_filter]
#     semi_true_obs_frames = ana_loc.size
# [
#         {
#             "loc_not_gaze": (loc_not_gaze := np.sum(ana_loc & ~ana_gaze))
#             / semi_true_obs_frames,
#             "gaze_not_loc": (gaze_not_loc := np.sum(ana_gaze & ~ana_loc))
#             / semi_true_obs_frames,
#             "observation_ratio": (observation_ratio := np.sum(ana_obs))
#             / semi_true_obs_frames,
#             "label": label,
#             "normalised": "semi_true_frames",
#         },
#         {
#             "loc_not_gaze": loc_not_gaze / frames,
#             "gaze_not_loc": gaze_not_loc / frames,
#             "observation_ratio": observation_ratio / frames,
#             "label": label,
#             "normalised": "frames",
#         },
#     ]
