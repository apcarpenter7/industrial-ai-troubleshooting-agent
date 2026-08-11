# module mcp_server

# system
import os
import json
import logging
from datetime import datetime
from typing import Any

# libs
from fastmcp import FastMCP

# import your existing underlying tool logic
from historian_tools import (
    tool_list_tags,
    tool_search_tags,
    tool_get_data_range,
    tool_get_tag_data,
    tool_get_statistics,
    tool_plot_tags,
    PLOTS_DIR,
)
from alarm_tools import (
    tool_alarm_query,
    tool_alarm_get_statistics,
    tool_alarm_get_active_at,
    tool_alarm_search_context,
    tool_alarm_detect_flood,
)
from kg_tools import (
    tool_kg_trace_stream,
    tool_kg_query_equipment,
    tool_kg_get_upstream_sensors,
    tool_kg_find_process_path,
    tool_kg_get_related_sensors,
    tool_kg_get_system_sensors,
)
from rag_tools import rag_search, rag_list_documents, rag_get_document
import audit_trace

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(
    filename=os.path.join(PROJECT_DIR, "historian_mcp.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastMCP Server Instance
# ---------------------------------------------------------------------------
mcp = FastMCP("Boiler Historian MCP Server")


# ── Historian Tools ────────────────────────────────────────────────────────

@mcp.tool()
def historian_list_tags() -> list[dict]:
    """List all available historian tags with descriptions, units, sensor type, and normal operating range."""
    audit_trace.ensure_session()
    return tool_list_tags()


@mcp.tool()
def historian_search_tags(query: str) -> list[dict]:
    """Search historian tags by keyword matching name, description, and sensor type."""
    audit_trace.ensure_session()
    return tool_search_tags(query)


@mcp.tool()
def historian_get_data_range() -> dict:
    """Return earliest and latest timestamp available in the historian database."""
    audit_trace.ensure_session()
    return tool_get_data_range()


@mcp.tool()
def historian_get_tag_data(tag_names: list[str], start_time: str, end_time: str, downsample_minutes: int = None) -> list[dict]:
    """Retrieve time-series data for one or more tags over a time range."""
    audit_trace.ensure_session()
    return tool_get_tag_data(tag_names, start_time, end_time, downsample_minutes)


@mcp.tool()
def historian_get_statistics(tag_names: list[str], start_time: str, end_time: str) -> dict:
    """Compute summary statistics (min, max, mean, std deviation, count) for tags over a time range."""
    audit_trace.ensure_session()
    return tool_get_statistics(tag_names, start_time, end_time)


@mcp.tool()
def historian_plot_tags(tag_names: list[str], start_time: str, end_time: str, title: str = None, downsample_minutes: int = None, show_normal_range: bool = False) -> str:
    """Generate a time-series plot for tags and save it as a PNG file, returning the file path."""
    audit_trace.ensure_session()
    return tool_plot_tags(tag_names, start_time, end_time, title, downsample_minutes, show_normal_range)


# ── Knowledge Graph Tools ──────────────────────────────────────────────────

@mcp.tool()
def kg_trace_stream(stream_name: str) -> list[dict]:
    """Trace the complete path of a process stream through the boiler."""
    audit_trace.ensure_session()
    return tool_kg_trace_stream(stream_name)


@mcp.tool()
def kg_query_equipment(query: str) -> list[dict]:
    """Find equipment nodes in the boiler by natural-language name."""
    audit_trace.ensure_session()
    return tool_kg_query_equipment(query)


# ── RAG Document Search Tools ──────────────────────────────────────────────

@mcp.tool()
def docs_search(query: str, doc_type: str = None, equipment_id: str = None, sensor_tag: str = None, top_k: int = 5) -> list[dict]:
    """Semantic search across all plant documentation (SOPs, datasheets, maintenance, etc.)."""
    audit_trace.ensure_session()
    return rag_search(query, doc_type=doc_type, equipment_id=equipment_id, sensor_tag=sensor_tag, top_k=top_k)


# ── Alarm Log Tools ────────────────────────────────────────────────────────

@mcp.tool()
def alarm_query(start_time: str, end_time: str, tag_names: list[str] = None, alarm_level: str = None, priority: str = None, state: str = "ANY", limit: int = 100) -> list[dict]:
    """Fetch DCS alarm events from the alarm log within a time window."""
    audit_trace.ensure_session()
    return tool_alarm_query(start_time, end_time, tag_names=tag_names, alarm_level=alarm_level, priority=priority, state=state, limit=limit)


@mcp.tool()
def alarm_search_context(timestamp: str, window_minutes: int = 30, tag_names: list[str] = None) -> list[dict]:
    """Find all alarms within +/- window_minutes of a focal timestamp."""
    audit_trace.ensure_session()
    return tool_alarm_search_context(timestamp, window_minutes=window_minutes, tag_names=tag_names)


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("FastMCP server starting up on port 8050")
    # This runs the server using Server-Sent Events (SSE) on the requested port
    mcp.run(transport='sse', port=8050)