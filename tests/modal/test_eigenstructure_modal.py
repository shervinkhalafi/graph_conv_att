"""Tests for eigenstructure CLI remote commands.

Test rationale:
    The eigenstructure CLI provides list-remote and download commands
    for interacting with Modal-deployed studies. These tests verify:
    1. CLI commands are properly defined and accept expected parameters
    2. Required arguments are enforced

    The dedicated Modal eigenstructure volume and functions have been
    removed in favour of routing through the generic ``execute_task()``
    pipeline with ``task_type="eigenstructure"``.

Invariants:
    - CLI commands correctly accept expected parameters
"""


class TestEigenstructureCLIRemoteCommands:
    """Tests for eigenstructure CLI list-remote and download commands.

    These tests verify CLI behavior by mocking Modal function calls.
    """

    def test_list_remote_cli_exists(self):
        """list-remote command should be defined in CLI."""
        from click.testing import CliRunner

        from tmgg.experiments.eigenstructure_study.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["list-remote", "--help"])

        assert result.exit_code == 0
        assert "List eigenstructure studies" in result.output

    def test_download_cli_exists(self):
        """download command should be defined in CLI."""
        from click.testing import CliRunner

        from tmgg.experiments.eigenstructure_study.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["download", "--help"])

        assert result.exit_code == 0
        assert "Download eigenstructure study" in result.output
        assert "--remote-path" in result.output
        assert "--local-path" in result.output


class TestDownloadCLILogic:
    """Tests for download command logic with mocked Modal functions."""

    def test_download_requires_remote_and_local_path(self):
        """Download command should require both remote and local path."""
        from click.testing import CliRunner

        from tmgg.experiments.eigenstructure_study.cli import main

        runner = CliRunner()

        # Missing local-path
        result = runner.invoke(main, ["download", "--remote-path", "test"])
        assert result.exit_code != 0

        # Missing remote-path
        result = runner.invoke(main, ["download", "--local-path", "/tmp/out"])
        assert result.exit_code != 0
