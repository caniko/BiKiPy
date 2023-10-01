from pydantic import Field

enclosure_field = Field(
    None,
    description="Perimeter defining the enclosure of the trial, used for "
    "excluding coordinates that are outside as they are most likely mistaken",
)

timestamp_index_field = Field(None, description="Timestamp index for coordinates in reader DataFrame")
