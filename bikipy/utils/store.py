from typing import Union, SupportsFloat, SupportsInt
from collections import UserDict


class RangeDict(UserDict):
    def __init__(self, class_dict, **kwargs):

        class_dict = {key: class_dict[key] for key in sorted(class_dict)}

        super().__init__(class_dict, **kwargs)

    def __getitem__(self, key: Union[SupportsFloat, SupportsInt]):
        try:
            return super().__getitem__(key)
        except KeyError:
            proto_key = None
            for d_key in self.keys():
                if d_key <= key:
                    proto_key = d_key
                elif proto_key and d_key > key:
                    return super().__getitem__(proto_key)

            if proto_key is not None:
                return super().__getitem__(proto_key)


if __name__ == "__main__":
    a = RangeDict({1: "can", 4: "dej", 10: "x"})
    print(a[4])
