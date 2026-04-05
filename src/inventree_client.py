"""InvenTree API client — direct REST calls to the InvenTree server.

This module talks to InvenTree's REST API directly via httpx.
When cli-anything-inventree is ready, we can swap in CLI subprocess calls
or keep this as a faster direct-API alternative.
"""

import logging
from typing import Any

import httpx

from config import settings

logger = logging.getLogger(__name__)

API_BASE = f"{settings.inventree_url}/api"
HEADERS = {
    "Authorization": f"Token {settings.inventree_api_token}",
    "Content-Type": "application/json",
}


async def _get(path: str, params: dict | None = None) -> Any:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{API_BASE}{path}", headers=HEADERS, params=params)
        resp.raise_for_status()
        return resp.json()


async def _post(path: str, data: dict | None = None) -> Any:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(f"{API_BASE}{path}", headers=HEADERS, json=data)
        resp.raise_for_status()
        return resp.json()


async def _patch(path: str, data: dict | None = None) -> Any:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.patch(f"{API_BASE}{path}", headers=HEADERS, json=data)
        resp.raise_for_status()
        return resp.json()


# --- Search & Query ---

async def search_parts(query: str, limit: int = 20) -> list[dict]:
    """Search parts by name/description/keywords."""
    return await _get("/part/", params={"search": query, "limit": limit})


async def get_part(part_id: int) -> dict:
    """Get a single part by ID."""
    return await _get(f"/part/{part_id}/")


async def list_stock(part_id: int | None = None, location_id: int | None = None, limit: int = 50) -> list[dict]:
    """List stock items, optionally filtered by part or location."""
    params: dict[str, Any] = {"limit": limit}
    if part_id is not None:
        params["part"] = part_id
    if location_id is not None:
        params["location"] = location_id
    return await _get("/stock/", params=params)


async def list_locations(parent: int | None = None) -> list[dict]:
    """List stock locations, optionally under a parent."""
    params = {}
    if parent is not None:
        params["parent"] = parent
    return await _get("/stock/location/", params=params)


async def get_location(location_id: int) -> dict:
    """Get a single location by ID."""
    return await _get(f"/stock/location/{location_id}/")


async def list_categories(parent: int | None = None) -> list[dict]:
    """List part categories, optionally under a parent."""
    params = {}
    if parent is not None:
        params["parent"] = parent
    return await _get("/part/category/", params=params)


# --- Create & Update ---

async def create_part(name: str, description: str, category_id: int, **kwargs) -> dict:
    """Create a new part."""
    data = {"name": name, "description": description, "category": category_id, **kwargs}
    return await _post("/part/", data=data)


async def create_stock_item(part_id: int, quantity: float, location_id: int, **kwargs) -> dict:
    """Create a stock item (add stock for a part at a location)."""
    data = {"part": part_id, "quantity": quantity, "location": location_id, **kwargs}
    return await _post("/stock/", data=data)


async def update_stock_quantity(stock_id: int, quantity: float) -> dict:
    """Update the quantity of a stock item."""
    return await _patch(f"/stock/{stock_id}/", data={"quantity": quantity})


async def move_stock(stock_id: int, location_id: int) -> dict:
    """Move a stock item to a different location."""
    return await _patch(f"/stock/{stock_id}/", data={"location": location_id})


async def create_location(name: str, description: str = "", parent: int | None = None) -> dict:
    """Create a new stock location."""
    data: dict[str, Any] = {"name": name, "description": description}
    if parent is not None:
        data["parent"] = parent
    return await _post("/stock/location/", data=data)


async def create_category(name: str, description: str = "", parent: int | None = None) -> dict:
    """Create a new part category."""
    data: dict[str, Any] = {"name": name, "description": description}
    if parent is not None:
        data["parent"] = parent
    return await _post("/part/category/", data=data)


# --- Reporting ---

async def get_inventory_summary() -> dict:
    """Get a high-level summary of the inventory."""
    parts = await _get("/part/", params={"limit": 1})
    stock = await _get("/stock/", params={"limit": 1})
    locations = await _get("/stock/location/", params={"limit": 1})
    categories = await _get("/part/category/", params={"limit": 1})

    return {
        "total_parts": parts.get("count", 0) if isinstance(parts, dict) else len(parts),
        "total_stock_items": stock.get("count", 0) if isinstance(stock, dict) else len(stock),
        "total_locations": locations.get("count", 0) if isinstance(locations, dict) else len(locations),
        "total_categories": categories.get("count", 0) if isinstance(categories, dict) else len(categories),
    }
