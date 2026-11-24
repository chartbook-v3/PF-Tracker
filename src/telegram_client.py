"""Secret-safe Telegram Bot API client with retry/backoff and strict redaction."""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Iterable

logger = logging.getLogger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org"
MAX_MESSAGE_LENGTH = 4096

_BOT_PATH_TOKEN_RE = re.compile(r"bot\d+:[A-Za-z0-9_-]+")
_RAW_TOKEN_RE = re.compile(r"\b\d{6,}:[A-Za-z0-9_-]{20,}\b")


class TelegramDeliveryError(RuntimeError):
    """Raised when sending a Telegram message fails (always secret-redacted)."""


def redact_secrets(text: str, secrets: Iterable[str] = ()) -> str:
    """Scrub Telegram bot tokens, numeric bot ID prefixes, and chat IDs from text."""
    if not text:
        return ""

    sanitized = _BOT_PATH_TOKEN_RE.sub("bot[REDACTED]", str(text))
    sanitized = _RAW_TOKEN_RE.sub("[REDACTED]", sanitized)

    candidates: set[str] = set()
    for secret in secrets:
        if not secret:
            continue
        cleaned = str(secret).strip()
        if not cleaned:
            continue
        candidates.add(cleaned)
        encoded = urllib.parse.quote(cleaned, safe="")
        if encoded and encoded != cleaned:
            candidates.add(encoded)
        if ":" in cleaned:
            prefix = cleaned.split(":", 1)[0].strip()
            if len(prefix) >= 4:
                candidates.add(prefix)
                candidates.add(f"bot{prefix}")

    for candidate in sorted(candidates, key=len, reverse=True):
        sanitized = sanitized.replace(candidate, "[REDACTED]")

    return sanitized


def _extract_retry_after(body: str, attempt: int) -> float:
    """Parse Telegram 429 retry_after seconds, falling back to exponential backoff."""
    try:
        payload = json.loads(body)
        retry_after = payload.get("parameters", {}).get("retry_after")
        if retry_after is not None:
            delay = float(retry_after)
            if delay > 0:
                return min(delay, 60.0)
    except (ValueError, TypeError, AttributeError):
        pass
    return float(min(2**attempt, 30))


def send_telegram_message(
    text: str,
    bot_token: str | None = None,
    chat_id: str | None = None,
    parse_mode: str = "HTML",
    disable_web_page_preview: bool = True,
    max_retries: int = 3,
    timeout: int = 20,
) -> dict[str, Any]:
    """Send an HTML message via Telegram Bot API with secret redaction and retries."""
    if len(text) > MAX_MESSAGE_LENGTH:
        raise TelegramDeliveryError(
            f"Message length {len(text)} exceeds Telegram limit of {MAX_MESSAGE_LENGTH}."
        )

    resolved_token = (
        bot_token if bot_token is not None else os.environ.get("TELEGRAM_BOT_TOKEN", "")
    ).strip()
    resolved_chat_id = (
        chat_id if chat_id is not None else os.environ.get("TELEGRAM_CHAT_ID", "")
    ).strip()

    if not resolved_token or not resolved_chat_id:
        raise TelegramDeliveryError(
            "Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID environment variable."
        )

    secret_values = (resolved_token, resolved_chat_id)
    url = f"{TELEGRAM_API_BASE}/bot{resolved_token}/sendMessage"
    payload = {
        "chat_id": resolved_chat_id,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": disable_web_page_preview,
    }
    encoded_body = json.dumps(payload).encode("utf-8")

    attempts = max(1, int(max_retries))
    last_error_msg = "Telegram delivery failed after retries."

    for attempt in range(1, attempts + 1):
        req = urllib.request.Request(
            url,
            data=encoded_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                raw_body = response.read().decode("utf-8", errors="replace")
            try:
                result = json.loads(raw_body)
            except ValueError as err:
                safe_msg = redact_secrets(
                    f"Invalid JSON response from Telegram API: {err}", secret_values
                )
                logger.error(safe_msg)
                raise TelegramDeliveryError(safe_msg) from None

            if not result.get("ok"):
                desc = result.get("description", "Unknown Telegram API error")
                safe_msg = redact_secrets(
                    f"Telegram API returned error: {desc}", secret_values
                )
                logger.error(safe_msg)
                raise TelegramDeliveryError(safe_msg) from None

            logger.info("Telegram report sent successfully.")
            msg_id = None
            if isinstance(result.get("result"), dict):
                msg_id = result["result"].get("message_id")
            return {"ok": True, "message_id": msg_id}

        except urllib.error.HTTPError as err:
            status_code = err.code
            error_body = ""
            try:
                error_body = err.read().decode("utf-8", errors="replace")
            except Exception:
                error_body = ""

            safe_detail = redact_secrets(
                f"HTTP {status_code}: {err.reason} {error_body}".strip(),
                secret_values,
            )

            if status_code == 429 and attempt < attempts:
                delay = _extract_retry_after(error_body, attempt)
                logger.warning(
                    "Telegram API rate limited (HTTP 429); retrying in %.1fs (attempt %d/%d).",
                    delay,
                    attempt,
                    attempts,
                )
                time.sleep(delay)
                last_error_msg = safe_detail
                continue

            if 500 <= status_code < 600 and attempt < attempts:
                delay = float(min(2**attempt, 30))
                logger.warning(
                    "Transient Telegram server error (%s); retrying in %.1fs (attempt %d/%d).",
                    safe_detail,
                    delay,
                    attempt,
                    attempts,
                )
                time.sleep(delay)
                last_error_msg = safe_detail
                continue

            logger.error("Telegram delivery failed: %s", safe_detail)
            raise TelegramDeliveryError(f"Telegram delivery failed: {safe_detail}") from None

        except urllib.error.URLError as err:
            safe_detail = redact_secrets(
                f"Network error contacting Telegram API: {err.reason}",
                secret_values,
            )
            if attempt < attempts:
                delay = float(min(2**attempt, 30))
                logger.warning(
                    "%s; retrying in %.1fs (attempt %d/%d).",
                    safe_detail,
                    delay,
                    attempt,
                    attempts,
                )
                time.sleep(delay)
                last_error_msg = safe_detail
                continue

            logger.error("Telegram delivery failed: %s", safe_detail)
            raise TelegramDeliveryError(f"Telegram delivery failed: {safe_detail}") from None

        except TelegramDeliveryError:
            raise

        except Exception as err:
            safe_detail = redact_secrets(
                f"Unexpected error sending Telegram report: {err}",
                secret_values,
            )
            logger.error("Telegram delivery failed: %s", safe_detail)
            raise TelegramDeliveryError(safe_detail) from None

    logger.error("Telegram delivery failed: %s", last_error_msg)
    raise TelegramDeliveryError(last_error_msg)
