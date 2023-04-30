from functools import cached_property
from logging import getLogger
from pathlib import Path
from typing import Any, Optional, TypeVar, Type, ClassVar, TYPE_CHECKING, TypeVarTuple, Generic

import numpy as np
import pandas as pd
from pydantic.generics import GenericModel
from pydantic_numpy.dtype import NDArrayBool

from bikipy.core.base_class import BaseBikipyInspectMixin
from bikipy.core.typing import Label
from bikipy.core.video import VideoMetadataMixin
from bikipy.feature.attention.model import AttentionModelMixin
from bikipy.feature.physical_object.component.abc import QualiaComponentType, QualiaComponent
from bikipy.feature.tolerance.single import single_node_tolerance_model
from bikipy.perimeter.base import SinglePerimeter
from bikipy.reader.base import Reader
from bikipy.utils.plot.inspect import generic_inspection_finalization
from bikipy.feature.physical_object.analysis import PhysicalObjectSet

logger = getLogger(__name__)


class PhysicalObjectObservationQualia(BaseBikipyInspectMixin, VideoMetadataMixin, AttentionModelMixin):
    observation_qualia_components: list[QualiaComponent, ...] = ...
    perimeter_label: str = ...

    # Inspection fields
    trial_obj_label: Optional[Label]
    _fig: Any = None
    _axes: Any = None
    _exporting_figure: bool = False

    category = "physical_object"

    def __post_init__(self):
        self._fig, self._axes = self.video.subplots(
            nrows=self.number_of_components, ncols=self.max_component_row_length
        )
        self._fig.suptitle(
            "Observation cumulative filtration analysis",
            fontsize=self.video.upscaled_video.plotting_title_font_size * 1.1,
        )

    @cached_property
    def merged_components(self) -> NDArrayBool:
        result = np.logical_or.reduce([component.boolean_index for component in self.observation_components])
        self.generic_result_plotter(result, self.summary_axes_row[0], "CombinedComponents")
        return result

    @cached_property
    def post_tolerance_modeled_merged_components(self) -> NDArrayBool:
        result = single_node_tolerance_model(self.merged_components, self.video.fps)
        self.generic_result_plotter(result, self.summary_axes_row[1], "PostToleranceModeledCombinedComponents")
        return result

    @cached_property
    def pre_tolerance_modeled_merged_components(self) -> NDArrayBool:
        result = np.logical_or.reduce(
            [component.tolerance_modeled_combined_sensation for component in self.observation_components]
        )
        self.generic_result_plotter(result, self.summary_axes_row[2], "ToleranceModeledCombinedComponents")
        return result

    @cached_property
    def pre_post_tolerance_modeled_merged_components(self) -> NDArrayBool:
        result = single_node_tolerance_model(self.pre_tolerance_modeled_merged_components, self.video.fps)
        self.generic_result_plotter(result, self.summary_axes_row[3], "DoubleToleranceModeledCombinedComponents")
        return result

    def _summary_indexer(self, data_labels: list[str], with_component_label: bool = True) -> pd.MultiIndex:
        if with_component_label:
            additive = self.component_label.capitalize()
            data_labels = [f"{additive}{label}" for label in data_labels]
        return pd.MultiIndex.from_product([[self.perimeter_label], data_labels])

    @property
    def summary(self) -> pd.Series:
        or_summary = pd.Series(
            [
                self.boolean_array_to_seconds(self.merged_components),
                self.boolean_array_to_seconds(self.post_tolerance_modeled_merged_components),
                self.boolean_array_to_seconds(self.pre_tolerance_modeled_merged_components),
                self.boolean_array_to_seconds(self.pre_post_tolerance_modeled_merged_components),
            ],
            index=pd.MultiIndex.from_product(
                [
                    [self.perimeter_label],
                    ["OR_Observe", "OR_PostTolObserve", "OR_PreTolObserve", "OR_PrePostTolObserve"],
                ]
            ),
        )
        component_summaries = [component.component_summary for component in self.observation_components]
        self.inspect_attention()
        return pd.concat([or_summary, *component_summaries])

    def inspect_attention(self) -> None:
        if isinstance(self.inspect_arg, Path):
            name = f"{self.inspect_arg.stem}_{self.label}.jpg"
            if self.trial_obj_label:
                name = f"{self.trial_obj_label}_{name}"
            generic_inspection_finalization(self.class_inspect_arg / name)
        else:
            generic_inspection_finalization(self.class_inspect_arg)

    @property
    def fig(self):
        if self._fig is not None:
            return self._fig
        self.__post_init__()
        return self._fig

    @property
    def axes(self):
        if self._axes is not None:
            return self._axes
        self.__post_init__()
        return self._axes

    @property
    def summary_axes_row(self):
        return self.axes[-1]

    @property
    def number_of_components(self) -> int:
        return len(self.observation_component_classes)

    @property
    def max_component_row_length(self) -> int:
        return max(obs_profile.inspection_row_length for obs_profile in self.observation_component_classes)

    @property
    def _first_reader(self) -> Reader:
        return self.observation_components[0].reader

    def generic_result_plotter(self, valid_boolean_index: NDArrayBool, ax: Any, label: str) -> None:
        ax.set_title(label, fontsize=self.video.upscaled_video.plotting_title_font_size)
        ax.scatter(
            *self._first_reader.plot_prepared_kinematic_coordinates[valid_boolean_index].T, marker="x", color="green"
        )
