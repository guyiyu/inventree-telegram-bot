"""Inventory context snapshot for the system prompt.

Fetches categories, locations, and stock counts from InvenTree and writes
a human-readable snapshot to data/context.txt. This is injected into the
system prompt so the model starts each conversation with a warm cache —
spending tokens (plentiful) instead of API rounds (scarce).

Context is refreshed:
- Once on startup
- After any inventory write operation (create/update/move)
- After a conversation compaction event
NOT on a periodic timer (that would waste daily request quota).
"""

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import inventree_client as inv

from config import settings

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
SAMPLE_DIR = Path(__file__).parent.parent / "sample-data"
CONTEXT_FILE = DATA_DIR / "context.txt"
PROMPT_FILE = DATA_DIR / "prompt.txt"


def init_data_dir() -> None:
    """Ensure data/ exists and seed missing files from sample-data/."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if SAMPLE_DIR.exists():
        for sample_file in SAMPLE_DIR.iterdir():
            target = DATA_DIR / sample_file.name
            if not target.exists():
                shutil.copy2(sample_file, target)
                logger.info("Seeded %s from sample-data/", sample_file.name)


def get_prompt() -> str:
    """Read the system prompt from data/prompt.txt."""
    if PROMPT_FILE.exists():
        return PROMPT_FILE.read_text(encoding="utf-8")
    return "You are a helpful home inventory assistant."


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
        f"SITE_URL: {settings.inventree_site_url}",
        "",
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
    logger.info("Context refreshed (%d bytes)", len(context))
