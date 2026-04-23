"""
Tests for the LoRA trainer launcher.

subprocess.create_subprocess_exec is mocked — no real processes spawned.
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_settings(deepseek="deepseek-ai/ds", mistral="mistralai/ms", output_dir="/tmp/lora", hf_token=""):
    s = MagicMock()
    s.deepseek_model = deepseek
    s.mistral_model = mistral
    s.output_dir = output_dir
    s.hf_token = hf_token
    return s


def _make_proc():
    """Mock async subprocess that streams no output lines."""
    async def _empty_stdout():
        return
        yield  # makes this an async generator

    proc = MagicMock()
    proc.wait = AsyncMock(return_value=None)
    proc.returncode = 0
    proc.stdout = _empty_stdout()
    return proc


@pytest.mark.asyncio
class TestTrainerLauncher:
    async def test_launch_training_spawns_subprocess(self):
        from services.training.app.utils import trainer_launcher
        trainer_launcher._training_running = False

        mock_proc = _make_proc()
        mock_settings = _make_settings()

        with (
            patch("services.training.app.utils.trainer_launcher.settings", mock_settings),
            patch(
                "asyncio.create_subprocess_exec",
                new_callable=AsyncMock,
                return_value=mock_proc,
            ) as mock_exec,
        ):
            await trainer_launcher.launch_training(
                capability="coding",
                training_records=[{"problem": "p", "solution": "s", "evaluation": 0.8}],
            )

        mock_exec.assert_called_once()
        # First positional arg is the executable
        cmd_args = mock_exec.call_args[0]
        assert "trainer_entry" in " ".join(str(a) for a in cmd_args)

    async def test_coding_capability_uses_deepseek_model(self):
        from services.training.app.utils import trainer_launcher
        trainer_launcher._training_running = False

        mock_proc = _make_proc()
        mock_settings = _make_settings(deepseek="deepseek-model", mistral="mistral-model")

        with (
            patch("services.training.app.utils.trainer_launcher.settings", mock_settings),
            patch(
                "asyncio.create_subprocess_exec",
                new_callable=AsyncMock,
                return_value=mock_proc,
            ) as mock_exec,
        ):
            await trainer_launcher.launch_training(
                capability="coding",
                training_records=[{"problem": "x", "solution": "y", "evaluation": 0.9}],
            )

        cmd_args = list(mock_exec.call_args[0])
        cmd_str = " ".join(str(a) for a in cmd_args)
        assert "deepseek-model" in cmd_str

    async def test_planning_capability_uses_mistral_model(self):
        from services.training.app.utils import trainer_launcher
        trainer_launcher._training_running = False

        mock_proc = _make_proc()
        mock_settings = _make_settings(deepseek="deepseek-model", mistral="mistral-model")

        with (
            patch("services.training.app.utils.trainer_launcher.settings", mock_settings),
            patch(
                "asyncio.create_subprocess_exec",
                new_callable=AsyncMock,
                return_value=mock_proc,
            ) as mock_exec,
        ):
            await trainer_launcher.launch_training(
                capability="planning",
                training_records=[{"problem": "x", "solution": "y", "evaluation": 0.9}],
            )

        cmd_str = " ".join(str(a) for a in mock_exec.call_args[0])
        assert "mistral-model" in cmd_str

    async def test_skips_launch_when_already_running(self):
        from services.training.app.utils import trainer_launcher
        trainer_launcher._training_running = True  # simulate running job

        with patch(
            "asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
        ) as mock_exec:
            await trainer_launcher.launch_training(
                capability="coding",
                training_records=[{"x": 1}],
            )

        mock_exec.assert_not_called()
        trainer_launcher._training_running = False  # cleanup

    async def test_skips_launch_when_no_records(self):
        from services.training.app.utils import trainer_launcher
        trainer_launcher._training_running = False

        with patch(
            "asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
        ) as mock_exec:
            await trainer_launcher.launch_training(capability="coding", training_records=[])

        mock_exec.assert_not_called()
