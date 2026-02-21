"""
Langfuse sync utilities for replaying logged telemetry events.

This module provides functionality to read telemetry events from a log file
and sync them to a Langfuse server using the REST API for proper historical replay.
"""
import json
import logging
import requests
from pathlib import Path
from typing import Optional
from datetime import datetime
from collections import defaultdict
import base64

logger = logging.getLogger(__name__)


class LangfuseOTLPClient:
    """Client for syncing telemetry data to Langfuse via REST API."""

    def __init__(self, langfuse_host: str, public_key: str, secret_key: str):
        """
        Initialize the Langfuse client.

        Args:
            langfuse_host: Langfuse server URL
            public_key: Langfuse public key
            secret_key: Langfuse secret key
        """
        self.langfuse_host = langfuse_host.rstrip('/')
        self.public_key = public_key
        self.secret_key = secret_key
        self.ingestion_url = f"{self.langfuse_host}/api/public/ingestion"
        self.batch = []

    def connect(self):
        """Verify connection to Langfuse server."""
        try:
            # Test connection
            response = requests.get(f"{self.langfuse_host}/api/public/health", timeout=5)
            if response.status_code == 200:
                logger.info(f"Connected to Langfuse at {self.langfuse_host}")
            else:
                logger.warning(f"Langfuse responded with status {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to connect to Langfuse: {e}")
            raise

    def send_event(self, event_type: str, event_data: dict, trace_id: str = None,
                   start_time: str = None, end_time: str = None):
        """
        Add event to batch for Langfuse ingestion.

        Args:
            event_type: Type of event (trace, span, generation, etc.)
            event_data: Event data dictionary
            trace_id: Trace ID for grouping
            start_time: ISO timestamp when event started
            end_time: ISO timestamp when event ended
        """
        try:
            # Extract common fields
            name = event_data.get("name", "unknown")
            metadata = event_data.get("metadata", {})
            input_val = event_data.get("input")
            output_val = event_data.get("output")

            # Build the event payload based on type
            if event_type == "trace":
                payload = {
                    "id": trace_id,
                    "type": "trace-create",
                    "body": {
                        "id": trace_id,
                        "name": name,
                        "input": input_val,
                        "output": output_val,
                        "metadata": metadata,
                        "timestamp": start_time,
                    }
                }

            elif event_type == "span":
                payload = {
                    "id": f"{trace_id}_span_{name}",
                    "type": "span-create",
                    "body": {
                        "id": f"{trace_id}_span_{name}",
                        "traceId": trace_id,
                        "name": name,
                        "startTime": start_time,
                        "endTime": end_time,
                        "metadata": metadata,
                        "input": input_val,
                        "output": output_val,
                    }
                }

            elif event_type == "generation":
                model = event_data.get("model", "unknown")
                usage = event_data.get("usage", {})

                payload = {
                    "id": f"{trace_id}_gen_{name}",
                    "type": "generation-create",
                    "body": {
                        "id": f"{trace_id}_gen_{name}",
                        "traceId": trace_id,
                        "name": name,
                        "model": model,
                        "startTime": start_time,
                        "endTime": end_time,
                        "metadata": metadata,
                        "input": input_val,
                        "output": output_val,
                        "usage": usage if usage else None,
                    }
                }

            else:
                logger.warning(f"Unknown event type: {event_type}")
                return False

            # Add to batch
            self.batch.append(payload)
            return True

        except Exception as e:
            logger.error(f"Failed to prepare {event_type} event: {e}")
            logger.debug(f"Event data: {event_data}")
            return False

    def flush(self):
        """Send batched events to Langfuse ingestion API."""
        if not self.batch:
            return

        try:
            # Prepare auth header
            auth = base64.b64encode(f"{self.public_key}:{self.secret_key}".encode()).decode()
            headers = {
                "Authorization": f"Basic {auth}",
                "Content-Type": "application/json",
            }

            # Send batch
            payload = {
                "batch": self.batch,
                "metadata": {
                    "sdk_name": "python",
                    "sdk_version": "sync_script",
                }
            }

            response = requests.post(
                self.ingestion_url,
                json=payload,
                headers=headers,
                timeout=10
            )

            if response.status_code in [200, 201, 207]:
                logger.info(f"Successfully flushed {len(self.batch)} events")
                self.batch = []  # Clear batch
            else:
                logger.error(f"Failed to flush events: HTTP {response.status_code}")
                logger.debug(f"Response: {response.text}")

        except Exception as e:
            logger.error(f"Failed to flush batch: {e}")


class LangfuseSyncService:
    """Service for syncing log files to Langfuse."""

    def __init__(self, langfuse_client: LangfuseOTLPClient, batch_size: int = 100):
        """
        Initialize the sync service.

        Args:
            langfuse_client: Langfuse client instance
            batch_size: Number of events to buffer before flushing
        """
        self.client = langfuse_client
        self.batch_size = batch_size

    def sync_log_file(self, log_file: Path) -> tuple[int, int]:
        """
        Sync events from a log file to Langfuse.

        Args:
            log_file: Path to the log file containing telemetry events

        Returns:
            Tuple of (total_events, successful_events)
        """
        if not log_file.exists():
            logger.error(f"Log file not found: {log_file}")
            return 0, 0

        # Connect to Langfuse
        try:
            self.client.connect()
        except Exception as e:
            logger.error(f"Failed to connect to Langfuse: {e}")
            return 0, 0

        total_events = 0
        successful_events = 0
        batch_count = 0

        logger.info(f"Reading events from {log_file}")

        with open(log_file, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    event = json.loads(line)
                    event_type = event.get("event_type")
                    event_data = event.get("data", {})
                    trace_id = event.get("trace_id")
                    start_time = event.get("start_time")
                    end_time = event.get("end_time")

                    total_events += 1

                    if self.client.send_event(event_type, event_data, trace_id, start_time, end_time):
                        successful_events += 1
                        batch_count += 1

                        # Flush in batches
                        if batch_count >= self.batch_size:
                            logger.info(f"Flushing batch of {batch_count} events...")
                            self.client.flush()
                            batch_count = 0

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON at line {line_num}: {e}")
                except Exception as e:
                    logger.error(f"Failed to process event at line {line_num}: {e}")

        # Flush remaining events
        if batch_count > 0:
            logger.info(f"Flushing final batch of {batch_count} events...")
            self.client.flush()

        logger.info(
            f"Sync complete: {successful_events}/{total_events} events synced successfully"
        )

        return total_events, successful_events
