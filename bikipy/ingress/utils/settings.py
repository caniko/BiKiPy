from typing import Optional


def get_definable_settings(root_dict: dict) -> set:
    definable_fields = set()
    if "optional" in root_dict:
        definable_fields.update(root_dict["optional"])
    if "required" in root_dict:
        definable_fields.update(root_dict["required"])
    return definable_fields


def update_defined_values(
    old_values: dict, new_values: dict, common_settings: Optional[set] = None, delete_outdated: bool = False
) -> dict:
    old_defined = old_values["defined"]

    old_defined_fields = set(old_defined)
    new_definable_fields = get_definable_settings(new_values)
    if common_settings:
        new_definable_fields.update(common_settings)

    fields_to_keep = old_defined_fields.intersection(new_definable_fields)

    for sub_field, sub_value in old_defined.items():
        if sub_field in fields_to_keep:
            new_values["defined"][sub_field] = sub_value
        elif not delete_outdated:
            if "outdated" not in new_values["defined"]:
                new_values["defined"]["outdated"] = {}
            new_values["defined"]["outdated"][sub_field] = sub_value

    return new_values


def update_dictionary(old: dict, new: dict, delete_outdated: bool = False) -> dict:
    outdated_fields = set(old).difference(new)
    if outdated_fields and not delete_outdated:
        new["outdated"] = {}

    for field, value in old.items():
        if field in new:
            if isinstance(value, dict):
                if field == "defined":
                    update_defined_dictionary(value, new[field], new["optional"], delete_outdated)
                else:
                    update_dictionary(value, new[field], delete_outdated)
            else:
                new[field] = value
        elif not delete_outdated:
            new["outdated"][field] = value

    return new


def update_defined_dictionary(old: dict, new: dict, new_optional: dict, delete_outdated: bool = False) -> dict:
    new_required_fields = set(old).difference(new)
    outdated_fields = new_required_fields.difference(new_optional)
    if outdated_fields and not delete_outdated:
        new["outdated"] = {}

    for field, value in old.items():
        if field in new or field in new_optional:
            new[field] = value
        elif not delete_outdated:
            new["outdated"][field] = value

    return new
