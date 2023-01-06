from bikipy.reader.base import ReaderCLS
from bikipy.reader.data_with_likelihood import DataWithLikelihoodReader


READERS: set[ReaderCLS] = {DataWithLikelihoodReader}

_READER_CLASS_NAME_TO_CLASS: dict[str, ReaderCLS] = {
    reader.__name__: reader for reader in READERS
}
READER_CLASS_LABEL_TO_CLASS: dict[str, ReaderCLS] = {
    **_READER_CLASS_NAME_TO_CLASS,
    "DeepLabCutReader": DataWithLikelihoodReader
}
