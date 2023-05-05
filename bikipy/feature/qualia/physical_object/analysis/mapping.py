from pydantic import PositiveInt

from bikipy.feature.qualia.physical_object.analysis.i import (
    OnePhysicalObjectSetQualiaAnalysis,
    QualiaAnalysis,
)
from bikipy.feature.qualia.physical_object.analysis.ii import (
    TwoPhysicalObjectSetQualiaAnalysis,
)
from bikipy.feature.qualia.physical_object.analysis.iii import (
    ThreePhysicalObjectSetQualiaAnalysis,
)
from bikipy.feature.qualia.physical_object.analysis.iv import (
    FourPhysicalObjectSetQualiaAnalysis,
)

PO_NUMBER_TO_ANALYSIS_MODEL: dict[PositiveInt, QualiaAnalysis] = {
    idx: model
    for idx, model in enumerate(
        (
            OnePhysicalObjectSetQualiaAnalysis,
            TwoPhysicalObjectSetQualiaAnalysis,
            ThreePhysicalObjectSetQualiaAnalysis,
            FourPhysicalObjectSetQualiaAnalysis,
        ),
        start=1,
    )
}
