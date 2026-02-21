"""
Telemetry logger that stores Langfuse events to a log file for later syncing.
"""
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .config import LOGS_DIR

# Telemetry log file (separate from Python logger output)
TELEMETRY_LOG = LOGS_DIR / "telemetry.jsonl"

logger = logging.getLogger(__name__)


def log_telemetry_event(event_type: str, event_data: dict[str, Any],
                        trace_id: Optional[str] = None,
                        start_time: Optional[str] = None,
                        end_time: Optional[str] = None):
    """
    Log a telemetry event to telemetry.jsonl in JSON format.

    Args:
        event_type: Type of event (trace, span, generation, etc.)
        event_data: Event data to log
        trace_id: Optional trace ID for grouping events
        start_time: ISO format start timestamp
        end_time: ISO format end timestamp
    """
    now = datetime.now().isoformat()

    event = {
        "timestamp": now,
        "event_type": event_type,
        "trace_id": trace_id or str(uuid.uuid4()),
        "start_time": start_time or now,
        "end_time": end_time or now,
        "data": event_data,
    }

    try:
        with open(TELEMETRY_LOG, "a") as f:
            f.write(json.dumps(event) + "\n")
        logger.debug(f"Logged {event_type} event to {TELEMETRY_LOG}")
    except Exception as e:
        logger.error(f"Failed to log telemetry event: {e}")


def log_trace(name: str, input_data: Any = None, output_data: Any = None,
              metadata: dict = None, trace_id: str = None,
              start_time: str = None, end_time: str = None, **kwargs):
    """Log a trace event."""
    event_data = {
        "name": name,
        "input": input_data,
        "output": output_data,
        "metadata": metadata or {},
        **kwargs,
    }
    log_telemetry_event("trace", event_data, trace_id=trace_id,
                       start_time=start_time, end_time=end_time)


def log_span(name: str, trace_id: str = None, metadata: dict = None,
            start_time: str = None, end_time: str = None, **kwargs):
    """Log a span event."""
    event_data = {
        "name": name,
        "metadata": metadata or {},
        **kwargs,
    }
    log_telemetry_event("span", event_data, trace_id=trace_id,
                       start_time=start_time, end_time=end_time)


def log_generation(name: str, model: str = None, input_data: Any = None, output_data: Any = None,
                   metadata: dict = None, usage: dict = None, trace_id: str = None,
                   start_time: str = None, end_time: str = None, **kwargs):
    """Log a generation event."""
    event_data = {
        "name": name,
        "model": model,
        "input": input_data,
        "output": output_data,
        "metadata": metadata or {},
        "usage": usage or {},
        **kwargs,
    }
    log_telemetry_event("generation", event_data, trace_id=trace_id,
                       start_time=start_time, end_time=end_time)
