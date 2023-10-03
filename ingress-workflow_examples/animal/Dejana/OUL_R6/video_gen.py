from bikipy.ingress.run import generate_inspection_videos

generate_inspection_videos(["1033_3"], ".", heuristics_to_use=["OlfactionHeuristic"], codec="av1_qsv")
