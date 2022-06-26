from functools import cached_property
from typing import ClassVar

from pydantic import FilePath

from bikipy.core.base_class import BaseBikipy


class BasePlugin(BaseBikipy):
    data_path: FilePath

    data_label: ClassVar[str]

    @cached_property
    def _info(self):
        return self.data_path.stem.split("-")

    @cached_property
    def sequence_index(self) -> int | None:
        first_split_data = self.data_path.stem.split(".")[0]
        if first_split_data.isdigit():
            return int(first_split_data)
