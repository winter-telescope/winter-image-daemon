# src/imagedaemon/utils/notify.py
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv  # pip install python-dotenv
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

log = logging.getLogger("imagedaemon.slack")


class SlackNotifier:
    """
    Tiny wrapper around slack_sdk.WebClient that loads
    credentials from environment variables or a .env file.
    """

    def __init__(self, *, env_file: str | Path | None = None):
        if env_file:
            load_dotenv(env_file)

        token = os.getenv("SLACK_BOT_TOKEN")
        channel = os.getenv("SLACK_CHANNEL")

        if not token or not channel:
            raise RuntimeError("SLACK_BOT_TOKEN and SLACK_CHANNEL must be set")

        self.client = WebClient(token=token)
        self.channel = channel

    # ------------------------------------------------------------------
    # simple wrappers
    # ------------------------------------------------------------------
    def post_text(self, text: str):
        try:
            self.client.chat_postMessage(channel=self.channel, text=text)
        except SlackApiError as e:
            log.warning("Slack post failed: %s", e)

    def post_image(self, image_path: str | Path, text: str | None = None):
        image_path = Path(image_path)
        if not image_path.exists():
            log.warning("Image %s does not exist; skipping Slack post", image_path)
            return

        try:
            self.client.files_upload(
                channels=self.channel,
                file=str(image_path),
                initial_comment=text or "",
                title=image_path.name,
            )
        except SlackApiError as e:
            log.warning("Slack image upload failed: %s", e)
