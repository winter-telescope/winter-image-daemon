# src/imagedaemon/utils/notify.py
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import requests
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

        token = os.getenv("SLACKBOT_TOKEN")
        channel = os.getenv("SLACK_CHANNEL")
        channel_id = os.getenv("SLACKBOT_CHANNEL_ID")

        if not token or not channel:
            raise RuntimeError("SLACKBOT_TOKEN and SLACK_CHANNEL must be set")

        self.client = WebClient(token=token)
        self.channel = channel
        self.channel_id = channel_id

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
            """
            # deprecated call to files_upload
            self.client.files_upload(
                channels=self.channel,
                file=str(image_path),
                initial_comment=text or "",
                title=image_path.name,
            )"""

            # new slack api call
            filename = os.path.basename(str(image_path))
            filesize = os.path.getsize(str(image_path))

            # Get upload URL
            upload_url_response = self.client.files_getUploadURLExternal(
                filename=filename, length=filesize
            )
            upload_url = upload_url_response["upload_url"]
            file_id = upload_url_response["file_id"]

            # Upload the file to the URL
            with open(str(image_path), "rb") as file_content:
                upload_response = requests.post(
                    upload_url,
                    data=file_content.read(),
                    headers={"Content-Type": "application/octet-stream"},
                )

            if upload_response.status_code != 200:
                raise RuntimeError(
                    f"Failed to upload image to Slack: {upload_response.status_code}"
                )

            # if we get here the file should have been uploaded successfully
            log.info("Image uploaded successfully to Slack")

            # Complete the upload
            complete_response = self.client.files_completeUploadExternal(
                files=[{"id": file_id, "title": image_path.name}],
                channel_id=self.channel_id,
                initial_comment=text or "",
            )

            log.info("Successfully uploaded image to channel")

        except SlackApiError as e:
            log.warning("Slack image upload failed: %s", e)

        except Exception as e:
            log.warning("Error uploading image to Slack: %s", e)
