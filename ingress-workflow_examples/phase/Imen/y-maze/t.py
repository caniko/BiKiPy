from bikipy.ingress.projectkit import ProjectKitJITBikipyConfiguration

projectkit_jit = ProjectKitJITBikipyConfiguration()

config = projectkit_jit.jit_config(
    ingress_method="phase", experiment_name="y_maze", qualia_heuristic=["OlfactionHeuristic", "ObjectInProximalFOV"]
)

# config.initialize()
config.update_and_dump()
