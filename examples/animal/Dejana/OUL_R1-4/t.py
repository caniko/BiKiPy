from bikipy.ingress.projectkit import ProjectKitJITBikipyConfiguration

projectkit_jit = ProjectKitJITBikipyConfiguration()

projectkit_jit.init_config(
    ingress_method="animal",
    experiment_name="oul",
    qualia_heuristic=["BodyProximity", "Olfaction", "WhiskerInteraction"],
)
