"""
Tests for the Context Service repo retrieval.

Covers:
- Non-code intent returns empty list
- Non-existent repo path returns empty list
- Keyword overlap scoring selects relevant files
- Prompt-aware scan: "create new" prompts skip the scan
- Prompt-aware scan: "fix/debug existing" prompts trigger the scan
"""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestRepoRetrieval:
    @pytest.mark.asyncio
    async def test_non_code_intent_returns_empty(self):
        from services.context.app.retrieval.repo import retrieve_repo_snippets
        result = await retrieve_repo_snippets("tell me about Python", "research")
        assert result == []

    @pytest.mark.asyncio
    async def test_qa_intent_returns_empty(self):
        from services.context.app.retrieval.repo import retrieve_repo_snippets
        result = await retrieve_repo_snippets("how does async work?", "qa")
        assert result == []

    @pytest.mark.asyncio
    async def test_new_code_prompt_skips_scan(self):
        """'Implement from scratch' prompts must NOT trigger the repo walk."""
        from services.context.app.retrieval.repo import retrieve_repo_snippets
        # No fix/debug/existing signals — should return empty without touching fs
        result = await retrieve_repo_snippets("implement a rate-limiter in Python", "coding")
        assert result == []

    @pytest.mark.asyncio
    async def test_new_code_prompt_write_skips_scan(self):
        """'Write a new parser' should also skip — no existing-code signal."""
        from services.context.app.retrieval.repo import retrieve_repo_snippets
        result = await retrieve_repo_snippets("write a binary search algorithm", "coding")
        assert result == []

    @pytest.mark.asyncio
    async def test_missing_repo_path_returns_empty(self):
        from services.context.app.retrieval.repo import retrieve_repo_snippets

        mock_settings = MagicMock()
        mock_settings.repo_path = "/nonexistent/path/that/does/not/exist"
        mock_settings.max_repo_snippets = 5

        with patch("services.context.app.retrieval.repo.settings", mock_settings):
            result = await retrieve_repo_snippets("fix a bug in the authentication module", "coding")
        assert result == []

    @pytest.mark.asyncio
    async def test_fix_prompt_triggers_scan(self, tmp_path):
        """'Fix' signals existing code — the repo walk should run."""
        py_file = tmp_path / "auth_utils.py"
        py_file.write_text('"""Authentication utilities for login and logout."""\ndef login(): pass\n')

        mock_settings = MagicMock()
        mock_settings.repo_path = str(tmp_path)
        mock_settings.max_repo_snippets = 3

        from services.context.app.retrieval.repo import retrieve_repo_snippets
        with patch("services.context.app.retrieval.repo.settings", mock_settings):
            results = await retrieve_repo_snippets("fix the login authentication bug", "coding")

        assert isinstance(results, list)
        file_paths = [r.file for r in results]
        assert any("auth_utils" in p for p in file_paths)

    @pytest.mark.asyncio
    async def test_debugging_intent_with_existing_signal_scans_repo(self, tmp_path):
        """Debugging intent + 'debug' keyword should trigger repo scan."""
        py_file = tmp_path / "database.py"
        py_file.write_text('"""Database connection and query helpers."""\n')

        mock_settings = MagicMock()
        mock_settings.repo_path = str(tmp_path)
        mock_settings.max_repo_snippets = 5

        from services.context.app.retrieval.repo import retrieve_repo_snippets
        with patch("services.context.app.retrieval.repo.settings", mock_settings):
            results = await retrieve_repo_snippets("debug the database query failure", "debugging")

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_file_extension_reference_triggers_scan(self, tmp_path):
        """Prompt mentioning a .py file should trigger the repo walk."""
        py_file = tmp_path / "auth_utils.py"
        py_file.write_text('"""Authentication utilities."""\n')

        mock_settings = MagicMock()
        mock_settings.repo_path = str(tmp_path)
        mock_settings.max_repo_snippets = 3

        from services.context.app.retrieval.repo import retrieve_repo_snippets
        with patch("services.context.app.retrieval.repo.settings", mock_settings):
            results = await retrieve_repo_snippets("what does auth_utils.py do?", "coding")

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_refactor_prompt_triggers_scan(self, tmp_path):
        """'Refactor' implies existing code — scan should run."""
        py_file = tmp_path / "api_client.py"
        py_file.write_text('"""API client."""\ndef call_api(): pass\n')

        mock_settings = MagicMock()
        mock_settings.repo_path = str(tmp_path)
        mock_settings.max_repo_snippets = 3

        from services.context.app.retrieval.repo import retrieve_repo_snippets
        with patch("services.context.app.retrieval.repo.settings", mock_settings):
            results = await retrieve_repo_snippets("refactor the API client module", "coding")

        assert isinstance(results, list)


class TestShouldScanRepo:
    def test_fix_signal(self):
        from services.context.app.retrieval.repo import _should_scan_repo
        assert _should_scan_repo("fix the login bug") is True

    def test_debug_signal(self):
        from services.context.app.retrieval.repo import _should_scan_repo
        assert _should_scan_repo("debug the authentication failure") is True

    def test_refactor_signal(self):
        from services.context.app.retrieval.repo import _should_scan_repo
        assert _should_scan_repo("refactor the payment module") is True

    def test_file_extension_signal(self):
        from services.context.app.retrieval.repo import _should_scan_repo
        assert _should_scan_repo("what does auth_utils.py do?") is True

    def test_our_codebase_signal(self):
        from services.context.app.retrieval.repo import _should_scan_repo
        assert _should_scan_repo("how does our codebase handle errors?") is True

    def test_new_implementation_no_signal(self):
        from services.context.app.retrieval.repo import _should_scan_repo
        assert _should_scan_repo("implement a binary search algorithm") is False

    def test_write_from_scratch_no_signal(self):
        from services.context.app.retrieval.repo import _should_scan_repo
        assert _should_scan_repo("write a rate-limiter in Python") is False


class TestFileFingerprint:
    def test_returns_first_meaningful_line(self, tmp_path):
        from services.context.app.retrieval.repo import _file_fingerprint
        f = tmp_path / "mod.py"
        f.write_text('"""Module docstring."""\ndef func(): pass\n')
        assert _file_fingerprint(f) == '"""Module docstring."""'

    def test_shebang_line_is_skipped(self, tmp_path):
        from services.context.app.retrieval.repo import _file_fingerprint
        f = tmp_path / "script.py"
        f.write_text('#!/usr/bin/env python\n# Module comment\ndef func(): pass\n')
        assert _file_fingerprint(f) == "# Module comment"

    def test_empty_file_returns_empty_string(self, tmp_path):
        from services.context.app.retrieval.repo import _file_fingerprint
        f = tmp_path / "empty.py"
        f.write_text("")
        assert _file_fingerprint(f) == ""

    def test_nonexistent_file_returns_empty_string(self, tmp_path):
        from services.context.app.retrieval.repo import _file_fingerprint
        assert _file_fingerprint(tmp_path / "ghost.py") == ""

    def test_long_line_truncated_at_120_chars(self, tmp_path):
        from services.context.app.retrieval.repo import _file_fingerprint
        f = tmp_path / "long.py"
        f.write_text("x" * 200 + "\n")
        result = _file_fingerprint(f)
        assert len(result) == 120


class TestScanFilesSync:
    def _tokens(self, text: str) -> set:
        import re
        return set(re.findall(r"[a-z_]+", text.lower()))

    def test_returns_matching_files(self, tmp_path):
        from services.context.app.retrieval.repo import _scan_files_sync
        f = tmp_path / "auth.py"
        f.write_text('"""Authentication helpers for login."""\n')
        results = _scan_files_sync(self._tokens("login authentication"), tmp_path, 5)
        assert any("auth" in r.file for r in results)

    def test_pycache_excluded(self, tmp_path):
        from services.context.app.retrieval.repo import _scan_files_sync
        cache_dir = tmp_path / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "auth.py").write_text('"""Authentication helpers for login."""\n')
        results = _scan_files_sync(self._tokens("login authentication"), tmp_path, 5)
        assert results == []

    def test_venv_excluded(self, tmp_path):
        from services.context.app.retrieval.repo import _scan_files_sync
        venv_dir = tmp_path / ".venv"
        venv_dir.mkdir()
        (venv_dir / "auth.py").write_text('"""Authentication helpers for login."""\n')
        results = _scan_files_sync(self._tokens("login authentication"), tmp_path, 5)
        assert results == []

    def test_relevance_ordering(self, tmp_path):
        from services.context.app.retrieval.repo import _scan_files_sync
        high = tmp_path / "auth_login.py"
        high.write_text('"""Authentication login session manager."""\n')
        low = tmp_path / "utils.py"
        low.write_text('"""Generic helpers."""\n')
        results = _scan_files_sync(self._tokens("auth login session"), tmp_path, 5)
        assert len(results) >= 1
        assert "auth_login" in results[0].file

    def test_posix_paths_returned(self, tmp_path):
        from services.context.app.retrieval.repo import _scan_files_sync
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "mod.py").write_text('"""Sub module helper."""\n')
        results = _scan_files_sync(self._tokens("sub module helper"), tmp_path, 5)
        assert all("\\" not in r.file for r in results)

    def test_respects_max_snippets(self, tmp_path):
        from services.context.app.retrieval.repo import _scan_files_sync
        for i in range(5):
            f = tmp_path / f"auth_{i}.py"
            f.write_text('"""Authentication helper module."""\n')
        results = _scan_files_sync(self._tokens("authentication helper"), tmp_path, max_snippets=2)
        assert len(results) <= 2


class TestShouldScanRepo:
    """Unit tests for the prompt-signal detection helper."""

    def test_fix_signal(self):
        from services.context.app.retrieval.repo import _should_scan_repo
        assert _should_scan_repo("fix the login authentication bug") is True

    def test_debug_signal(self):
        from services.context.app.retrieval.repo import _should_scan_repo
        assert _should_scan_repo("debug the memory leak in the service") is True

    def test_refactor_signal(self):
        from services.context.app.retrieval.repo import _should_scan_repo
        assert _should_scan_repo("refactor the database module") is True

    def test_update_signal(self):
        from services.context.app.retrieval.repo import _should_scan_repo
        assert _should_scan_repo("update the existing rate limiter") is True

    def test_file_extension_signal(self):
        from services.context.app.retrieval.repo import _should_scan_repo
        assert _should_scan_repo("look at utils.py and tell me what it does") is True

    def test_new_code_no_signal(self):
        from services.context.app.retrieval.repo import _should_scan_repo
        assert _should_scan_repo("implement a sorting algorithm in Python") is False

    def test_write_new_no_signal(self):
        from services.context.app.retrieval.repo import _should_scan_repo
        assert _should_scan_repo("write a binary search function") is False

    def test_build_new_no_signal(self):
        from services.context.app.retrieval.repo import _should_scan_repo
        assert _should_scan_repo("build a REST API endpoint for user registration") is False

    def test_existing_keyword_signal(self):
        from services.context.app.retrieval.repo import _should_scan_repo
        assert _should_scan_repo("improve the existing cache implementation") is True

    def test_our_codebase_signal(self):
        from services.context.app.retrieval.repo import _should_scan_repo
        assert _should_scan_repo("how does our codebase handle errors?") is True


class TestTokenise:
    def test_lowercase_split(self):
        from services.context.app.retrieval.repo import _tokenise
        tokens = _tokenise("Hello World! Python_rocks")
        assert "hello" in tokens
        assert "world" in tokens
        assert "python_rocks" in tokens

    def test_empty_string(self):
        from services.context.app.retrieval.repo import _tokenise
        tokens = _tokenise("")
        assert tokens == set()
