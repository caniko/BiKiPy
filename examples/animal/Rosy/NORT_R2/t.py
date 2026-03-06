from bikipy.ingress.projectkit import ProjectKitJITBikipyConfiguration

projectkit_jit = ProjectKitJITBikipyConfiguration()

projectkit_jit.jit_config(
    ingress_method="animal",
    experiment_name="nort",
    qualia_heuristic=["BodyProximity", "Olfaction", "WhiskerInteraction"],
).initialize()
