"""Periodic compaction of inventory state into a context summary.

Fetches categories, locations, and stock counts from InvenTree and writes
a human-readable snapshot to context.txt. This is injected into the system
prompt so the model starts each conversation with a warm cache — spending
tokens (plentiful) instead of API rounds (scarce).
"""

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

import inventree_client as inv

from config import settings

logger = logging.getLogger(__name__)

CONTEXT_FILE = Path(__file__).parent / "context.txt"


def get_context() -> str:
    """Read the current context snapshot, or return empty if not yet generated."""
    if CONTEXT_FILE.exists():
        return CONTEXT_FILE.read_text(encoding="utf-8")
    return "(No inventory context available yet. Use functions to query.)"


async def _build_context() -> str:
    """Fetch inventory state and build a compact text summary."""
    try:
        categories = await inv.list_categories()
        locations = await inv.list_locations()
        summary = await inv.get_inventory_summary()
    except Exception:
        logger.exception("Failed to fetch inventory data for compaction")
        return get_context()  # keep the old one

    lines = [
        f"Inventory snapshot (UTC {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}):",
        f"  Total parts: {summary.get('total_parts', '?')}",
        f"  Total stock items: {summary.get('total_stock_items', '?')}",
        f"  Total locations: {summary.get('total_locations', '?')}",
        f"  Total categories: {summary.get('total_categories', '?')}",
        "",
        "Categories:",
    ]

    cat_list = categories if isinstance(categories, list) else categories.get("results", [])
    for cat in cat_list:
        parent_info = f" (under #{cat['parent']})" if cat.get("parent") else ""
        lines.append(f"  #{cat['pk']}: {cat['name']}{parent_info}")

    lines.append("")
    lines.append("Locations:")

    loc_list = locations if isinstance(locations, list) else locations.get("results", [])
    for loc in loc_list:
        parent_info = f" (under #{loc['parent']})" if loc.get("parent") else ""
        lines.append(f"  #{loc['pk']}: {loc['name']}{parent_info}")

    return "\n".join(lines)


async def refresh_context() -> None:
    """Rebuild and save the context snapshot."""
    context = await _build_context()
    CONTEXT_FILE.write_text(context, encoding="utf-8")
    logger.info("Context compacted (%d bytes)", len(context))


async def compaction_loop() -> None:
    """Run in the background, refreshing context periodically."""
    while True:
        await refresh_context()
        await asyncio.sleep(settings.compaction_interval)
