from pathlib import PurePath
from typing import Any, Generic, Literal, Mapping, Optional, TypeVar, Union, get_origin

import numpy as np
from pydantic import BaseModel, FilePath
from pydantic.fields import ModelField

Path_typing = Union[PurePath, str]
Path_typing_kwarg = Union[PurePath, str, None]

DType = TypeVar("DType")


class NPFileDesc(BaseModel):
    path: FilePath
    key: Optional[str]


class NDArray(np.ndarray, Generic[DType]):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, val: Any, field: ModelField):
        if isinstance(val, Mapping):
            val = NPFileDesc(**val)
        if isinstance(val, NPFileDesc):
            val: NPFileDesc
            path = val.path
            key = val.key
            if path.suffix.lower() not in [".npz", ".npy"]:
                raise ValueError("Expected npz or npy file.")

            content = np.load(str(val.path.absolute()))
            if path.suffix.lower() == ".npz":
                key = key or content.files[0]
                data = content[key]
            else:
                data = content
        else:
            data = val

        if field.sub_fields is not None:
            dtype_field = field.sub_fields[0]
            if not get_origin(dtype_field.type_) == Literal:
                raise ValueError("DType field is expected to be Literal[str]")
            actual_dtype_lit = dtype_field.type_.__args__[0]
            return np.array(data, dtype=actual_dtype_lit)
        else:
            return np.array(data)
