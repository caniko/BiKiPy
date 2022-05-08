from click.testing import CliRunner

from bikipy.cli.ingress import sequence
from tests.constant import SEQUENCE_NORT_EXAMPLE_PROJECT_PATH


def test_sequence():
    runner = CliRunner()
    result = runner.invoke(sequence, ["-e", "nort", "-p", str(SEQUENCE_NORT_EXAMPLE_PROJECT_PATH)])
    assert result.exit_code == 0
    assert "Debug mode is on" in result.output
    assert "Syncing" in result.output
