from bikipy.ingress.projectkit import ProjectKitJITBikipyConfiguration

projectkit_jit = ProjectKitJITBikipyConfiguration()

config = projectkit_jit.jit_config(
    ingress_method="phase", experiment_name="nort", qualia_heuristic=["OlfactionHeuristic", "WhiskerInteraction"]
)

# config.initialize()
config.update_and_dump()
