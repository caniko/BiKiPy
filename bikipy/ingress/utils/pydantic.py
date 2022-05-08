from typing import Iterable


def name_to_type(fields: Iterable, class_schema: dict):
    result = {}
    for name in sorted(fields):
        field_property = class_schema["properties"][name]
        if "type" in field_property:
            result[name] = field_property["type"]
        elif "anyOf" in field_property:
            result[name] = field_property["anyOf"]
        elif "allOf" in field_property:
            result[name] = field_property["allOf"]
        else:
            result[name] = None
    return result


def extended_schema(class_schema: dict, with_required: bool = True):
    required = name_to_type(class_schema["required"], class_schema)
    optional = name_to_type(
        set(class_schema["properties"]).difference(class_schema["required"]), class_schema
    )
    if with_required:
        return {
            "defined": dict.fromkeys(required),
            "required": required,
            "optional": optional,
        }

    return {
        "defined": None,
        "optional": optional
    }
