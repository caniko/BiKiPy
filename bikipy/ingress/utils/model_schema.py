from functools import reduce
from typing import Iterable

from pydantic import BaseModel

from bikipy.core.base_class import BaseBikipy
from bikipy.perimeter.base import Perimeter


def field_name_to_metadata(fields: Iterable, class_schema: dict, model_class: BaseBikipy) -> dict:
    result = {}
    for name in sorted(fields):
        if name in model_class.exclude_from_settings_schema:
            continue

        field_property = class_schema["properties"][name]
        if "type" in field_property:
            field_type = field_property["type"]
        elif "anyOf" in field_property:
            field_type = field_property["anyOf"]
        elif "allOf" in field_property:
            field_type = field_property["allOf"]
        else:
            field_type = None

        if "default" in field_property:
            result[name] = f"{field_type} -> {field_property['default']}"
        else:
            result[name] = str(field_type)

    return result


def extended_schema(model_class: BaseModel, with_optional: bool = True, with_required: bool = True) -> dict:
    assert with_optional or with_required

    model_class.update_forward_refs(Perimeter=Perimeter)
    class_schema = model_class.schema()
    required = (
        field_name_to_metadata(class_schema["required"], class_schema, model_class)
        if "required" in class_schema
        else {}
    )
    optional = field_name_to_metadata(set(class_schema["properties"]).difference(required), class_schema, model_class)

    defined, result = [], {}
    if required and with_required:
        defined.extend(required)
        result["required"] = required
    if with_optional:
        result["optional"] = optional

    return {"defined": dict.fromkeys(defined), **result}


def extended_group_schema(model_classes: Iterable, *args, **kwargs) -> dict:
    schemas = {model_class.__name__: extended_schema(model_class, *args, **kwargs) for model_class in model_classes}
    schema_names = tuple(schemas)
    first = schemas[schema_names[0]]

    specific, common = {}, {}
    for component in first:
        # Reduce till we have keys present in every set
        common_keys = set(reduce(set.intersection, map(lambda x: set(x[component]), schemas.values())))
        # Common keys are present in every schema; we can just get the metadata, v, from the first
        common[component] = {k: v for k, v in first[component].items() if k in common_keys}

        for name, schema in schemas.items():
            if difference := set(schema[component]).difference(common_keys):
                if name not in specific:
                    specific[name] = {}
                specific[name][component] = {k: v for k, v in schema[component].items() if k in difference}
            else:
                specific[name] = {"defined": {}}

    return {"common": common, "specific": specific}
