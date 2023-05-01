class Abstract:
    @cached_property
    def tolerance_modeled_combined_sensation(self) -> NDArrayBool:
        result = single_node_tolerance_model(self.boolean_index, self.video.fps)
        self.generic_result_plotter(result, self.axes_row[-1], "ToleranceModeledCombined")
        return result

    @cached_property
    def tolerance_modeled_combined_sensation_seconds(self) -> float:
        return self.boolean_array_to_seconds(self.tolerance_modeled_combined_sensation)

    @cached_property
    def tolerance_vs_unfiltered_ratio(self) -> float:
        return self.tolerance_modeled_combined_sensation_seconds / self.seconds_of_observation_qualia
