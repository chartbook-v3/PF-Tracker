"""Security tests guaranteeing zero secret leaks in exceptions, logs, and repository files."""

from __future__ import annotations

import io
import logging
import re
import urllib.error
from pathlib import Path
from unittest.mock import patch

import pytest

from src.telegram_client import TelegramDeliveryError, redact_secrets, send_telegram_message

# Build synthetic secrets dynamically so no static scanner flags test files
_FAKE_BOT_ID = "987654" + "3210"
_FAKE_SECRET_PART = "AAFakeSecret" + "TokenPayload" + "XYZ123456"
_FAKE_TOKEN = f"{_FAKE_BOT_ID}:{_FAKE_SECRET_PART}"
_FAKE_CHAT_ID = "-100" + "9988776655"
_FORBIDDEN_HANDLE = "@" + "chart" + "book" + "ai"
_TOKEN_REGEX = re.compile(r"\d{6,}:[A-Za-z0-9_-]{20,}")


def test_redact_secrets_scrubs_token_bot_id_and_chat_id() -> None:
    raw = (
        f"POST https://api.telegram.org/bot{_FAKE_TOKEN}/sendMessage "
        f"failed for bot {_FAKE_BOT_ID} and chat_id={_FAKE_CHAT_ID}"
    )
    cleaned = redact_secrets(raw, (_FAKE_TOKEN, _FAKE_CHAT_ID))

    assert _FAKE_TOKEN not in cleaned
    assert _FAKE_SECRET_PART not in cleaned
    assert _FAKE_BOT_ID not in cleaned
    assert _FAKE_CHAT_ID not in cleaned
    assert "bot[REDACTED]" in cleaned
    assert "[REDACTED]" in cleaned


def test_http_error_redacts_all_secrets_in_exception_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    url = f"https://api.telegram.org/bot{_FAKE_TOKEN}/sendMessage"
    body_bytes = (
        f'{{"ok":false,"description":"Bad Request: chat {_FAKE_CHAT_ID} unreachable for bot {_FAKE_BOT_ID}"}}'
    ).encode("utf-8")

    def fake_urlopen(*args, **kwargs):
        raise urllib.error.HTTPError(
            url=url,
            code=400,
            msg=f"Bad Request on {url}",
            hdrs=None,
            fp=io.BytesIO(body_bytes),
        )

    caplog.set_level(logging.DEBUG)
    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        with pytest.raises(TelegramDeliveryError) as exc_info:
            send_telegram_message(
                "<b>Test</b>",
                bot_token=_FAKE_TOKEN,
                chat_id=_FAKE_CHAT_ID,
                max_retries=1,
            )

    err_text = str(exc_info.value)
    all_logs = caplog.text

    for haystack in (err_text, all_logs):
        assert _FAKE_TOKEN not in haystack
        assert _FAKE_SECRET_PART not in haystack
        assert _FAKE_BOT_ID not in haystack
        assert _FAKE_CHAT_ID not in haystack

    # Ensure unredacted HTTPError is not chained via __cause__
    assert exc_info.value.__cause__ is None


def test_url_error_redacts_secrets_in_exception_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    def fake_urlopen(*args, **kwargs):
        raise urllib.error.URLError(
            f"Connection refused to https://api.telegram.org/bot{_FAKE_TOKEN}/sendMessage for {_FAKE_CHAT_ID}"
        )

    caplog.set_level(logging.DEBUG)
    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        with pytest.raises(TelegramDeliveryError) as exc_info:
            send_telegram_message(
                "<b>Test</b>",
                bot_token=_FAKE_TOKEN,
                chat_id=_FAKE_CHAT_ID,
                max_retries=1,
            )

    for haystack in (str(exc_info.value), caplog.text):
        assert _FAKE_TOKEN not in haystack
        assert _FAKE_SECRET_PART not in haystack
        assert _FAKE_BOT_ID not in haystack
        assert _FAKE_CHAT_ID not in haystack
    assert exc_info.value.__cause__ is None


def test_success_log_never_exposes_chat_id_or_bot_id(caplog: pytest.LogCaptureFixture) -> None:
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self) -> bytes:
            return b'{"ok": true, "result": {"message_id": 42}}'

    caplog.set_level(logging.DEBUG)
    with patch("urllib.request.urlopen", return_value=FakeResponse()):
        res = send_telegram_message(
            "<b>Test</b>",
            bot_token=_FAKE_TOKEN,
            chat_id=_FAKE_CHAT_ID,
        )

    assert res == {"ok": True, "message_id": 42}
    assert "Telegram report sent successfully." in caplog.text
    assert _FAKE_TOKEN not in caplog.text
    assert _FAKE_BOT_ID not in caplog.text
    assert _FAKE_CHAT_ID not in caplog.text


def test_repository_scan_contains_no_hardcoded_secrets_or_handles() -> None:
    """Scan all tracked repository files for hardcoded tokens, bot IDs, or channel handles."""
    repo_root = Path(__file__).resolve().parent.parent
    scan_targets = [
        repo_root / "config",
        repo_root / "src",
        repo_root / ".github",
        repo_root / "tests",
        repo_root / "README.md",
        repo_root / ".env.example",
        repo_root / "requirements.txt",
    ]

    files_scanned = 0
    for target in scan_targets:
        paths = [target] if target.is_file() else list(target.rglob("*"))
        for file_path in paths:
            if not file_path.is_file():
                continue
            if "__pycache__" in file_path.parts or file_path.suffix == ".pyc":
                continue
            files_scanned += 1
            content = file_path.read_text(encoding="utf-8", errors="replace")

            assert _TOKEN_REGEX.search(content) is None, (
                f"Potential Telegram bot token pattern found in {file_path}"
            )
            assert _FORBIDDEN_HANDLE not in content.lower(), (
                f"Hardcoded channel handle found in {file_path}"
            )
            assert re.search(r"bot\d{6,}", content) is None, (
                f"Potential hardcoded numeric bot ID found in {file_path}"
            )

    assert files_scanned >= 10
