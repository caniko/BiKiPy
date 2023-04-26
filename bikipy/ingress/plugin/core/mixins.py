from functools import cached_property
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field
from pydantic_numpy.dtype import NDArrayFp64

from bikipy.core.base_class import BikipyModel
from bikipy.ingress.workflow.base import BaseIngressWorkflow
from bikipy.utils.makesense import get_only_point_from_makesense


class IngressRequiredMixin(BaseModel):
    ingress: BaseIngressWorkflow = Field(
        ..., description="Bikipy ingress object to access project metadata relevant for defining perimeter"
    )


class HasReferenceMixin(BikipyModel):
    manual_reference: Optional[NDArrayFp64] = Field(
        description="Override the perimeter detection with values defined outside model"
    )

    @cached_property
    def reference_point(self) -> pd.DataFrame | None:
        if self.manual_reference is not None:
            return self.manual_reference
        if (path_to_reference_file := self.data_path.parent / f"reference-{self.stem_info.label}.csv").exists():
            return get_only_point_from_makesense(path_to_reference_file)


class TrialWiseMetadataOnlyMixin(BaseModel):
    def globally_defined(self) -> None:
        msg = f"{self.__class__.__name__} does not support globally defined"
        raise AttributeError(msg)
