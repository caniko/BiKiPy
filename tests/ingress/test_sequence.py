from bikipy.ingress.core import analyze
from bikipy.ingress.animal import sequence_generate_configuration
from tests.constant import SEQUENCE_NORT_EXAMPLE_PROJECT_PATH


def test_sequence_generate_configuration():
    settings = sequence_generate_configuration(SEQUENCE_NORT_EXAMPLE_PROJECT_PATH, "nort", dry_run=True)
    assert settings

    # There are two perimeter files in the Perimeter directory of this project
    assert len(settings["perimeter"]["info"]) == 2


def test_analyze_sequence():
    analysis_result = analyze(SEQUENCE_NORT_EXAMPLE_PROJECT_PATH)

    assert analysis_result
