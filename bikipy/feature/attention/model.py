from functools import cached_property
from typing import Any

from bikipy.core.base_class import BikipyModel


class AttentionModelMixin(BikipyModel):
    @cached_property
    def _global_attention_kwargs(self) -> dict[str, Any]:
        return {"inspect_video": self.video, "inspect_pixels": self.video.frame is not None}
