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


async def _delete(path: str, data: dict | None = None) -> bool:
    """Send a DELETE request. Returns True on success (204 No Content)."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.request("DELETE", f"{API_BASE}{path}", headers=HEADERS, json=data)
        resp.raise_for_status()
        return True


# --- Search & Query ---

async def search_parts(query: str, limit: int = 20) -> list[dict]:
    """Search parts by name/description/keywords."""
    return await _get("/part/", params={"search": query, "limit": limit})


async def get_part(part_id: int) -> dict:
    """Get a single part by ID."""
    return await _get(f"/part/{part_id}/")


async def get_stock_item(stock_id: int) -> dict:
    """Get a single stock item by ID."""
    return await _get(f"/stock/{stock_id}/")


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


async def get_category(category_id: int) -> dict:
    """Get a single category by ID."""
    return await _get(f"/part/category/{category_id}/")


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


async def move_location(location_id: int, parent: int) -> dict:
    """Move a location to a new parent (re-parent).

    All child locations and stock items move with it automatically.
    """
    return await _patch(f"/stock/location/{location_id}/", data={"parent": parent})


async def update_location(location_id: int, name: str | None = None, description: str | None = None) -> dict:
    """Update a location's name and/or description."""
    data: dict[str, Any] = {}
    if name is not None:
        data["name"] = name
    if description is not None:
        data["description"] = description
    if not data:
        return {"error": "No fields to update"}
    return await _patch(f"/stock/location/{location_id}/", data=data)


async def update_part(part_id: int, name: str | None = None, description: str | None = None,
                      category_id: int | None = None, keywords: str | None = None) -> dict:
    """Update a part's name, description, category, or keywords."""
    data: dict[str, Any] = {}
    if name is not None:
        data["name"] = name
    if description is not None:
        data["description"] = description
    if category_id is not None:
        data["category"] = category_id
    if keywords is not None:
        data["keywords"] = keywords
    if not data:
        return {"error": "No fields to update"}
    return await _patch(f"/part/{part_id}/", data=data)


async def update_category(category_id: int, name: str | None = None, description: str | None = None) -> dict:
    """Update a category's name and/or description."""
    data: dict[str, Any] = {}
    if name is not None:
        data["name"] = name
    if description is not None:
        data["description"] = description
    if not data:
        return {"error": "No fields to update"}
    return await _patch(f"/part/category/{category_id}/", data=data)


async def move_category(category_id: int, parent: int) -> dict:
    """Move a category to a new parent (re-parent).

    All child categories and parts stay nested under it.
    """
    return await _patch(f"/part/category/{category_id}/", data={"parent": parent})


async def add_stock(stock_id: int, quantity: float, notes: str = "") -> dict:
    """Add quantity to an existing stock item with tracking history.

    Uses the proper /api/stock/add/ endpoint which creates a stock tracking entry.
    """
    data: dict[str, Any] = {"items": [{"pk": stock_id, "quantity": quantity}]}
    if notes:
        data["notes"] = notes
    return await _post("/stock/add/", data=data)


async def remove_stock(stock_id: int, quantity: float, notes: str = "") -> dict:
    """Remove quantity from an existing stock item with tracking history.

    Uses the proper /api/stock/remove/ endpoint which creates a stock tracking entry.
    """
    data: dict[str, Any] = {"items": [{"pk": stock_id, "quantity": quantity}]}
    if notes:
        data["notes"] = notes
    return await _post("/stock/remove/", data=data)


async def transfer_stock(stock_id: int, quantity: float, location_id: int, notes: str = "") -> dict:
    """Transfer (split & move) a quantity of stock to a different location.

    If the transfer quantity equals the full stock item quantity, the item is
    simply moved. If less, InvenTree splits the stock item and moves the
    requested quantity to the new location. Creates tracking history.
    """
    data: dict[str, Any] = {
        "items": [{"pk": stock_id, "quantity": quantity}],
        "location": location_id,
    }
    if notes:
        data["notes"] = notes
    return await _post("/stock/transfer/", data=data)


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


# --- Parameters ---
# Parameters attach structured key-value attributes (weight, dimensions, price,
# expiry date, etc.) to Parts or Stock Locations via a template system.
# model_type is "part.part" for parts or "stock.stocklocation" for locations.

async def list_parameter_templates(limit: int = 100) -> Any:
    """List all available parameter templates (what attributes can be tracked)."""
    return await _get("/parameter/template/", params={"limit": limit})


async def create_parameter_template(name: str, units: str = "", description: str = "",
                                     checkbox: bool = False, choices: str = "") -> dict:
    """Create a new parameter template (define a new trackable attribute).

    Args:
        name: Parameter name (e.g. "Weight", "Purchase Price", "Warranty Expiry")
        units: Physical units (e.g. "kg", "EUR", "mm")
        description: Human-readable description
        checkbox: If True, this parameter is a boolean checkbox
        choices: Comma-separated valid choices (e.g. "Small,Medium,Large")
    """
    data: dict[str, Any] = {"name": name}
    if units:
        data["units"] = units
    if description:
        data["description"] = description
    if checkbox:
        data["checkbox"] = True
    if choices:
        data["choices"] = choices
    return await _post("/parameter/template/", data=data)


async def list_parameters(part_id: int | None = None, location_id: int | None = None,
                           template_id: int | None = None, limit: int = 100) -> Any:
    """List parameter values, optionally filtered by part, location, or template.

    Exactly one of part_id or location_id should be provided to get parameters
    for a specific object. template_id can narrow by parameter type.
    """
    params: dict[str, Any] = {"limit": limit}
    if part_id is not None:
        params["model_type"] = "part.part"
        params["model_id"] = part_id
    elif location_id is not None:
        params["model_type"] = "stock.stocklocation"
        params["model_id"] = location_id
    if template_id is not None:
        params["template"] = template_id
    return await _get("/parameter/", params=params)


async def set_parameter(template_id: int, data_value: str,
                         part_id: int | None = None, location_id: int | None = None,
                         note: str = "") -> dict:
    """Set a parameter value on a part or location.

    Args:
        template_id: Which parameter template to use
        data_value: The value to set (always a string; InvenTree parses units)
        part_id: Part ID to attach to (mutually exclusive with location_id)
        location_id: Location ID to attach to (mutually exclusive with part_id)
        note: Optional note about this parameter value
    """
    data: dict[str, Any] = {"template": template_id, "data": data_value}
    if part_id is not None:
        data["model_type"] = "part.part"
        data["model_id"] = part_id
    elif location_id is not None:
        data["model_type"] = "stock.stocklocation"
        data["model_id"] = location_id
    else:
        return {"error": "Either part_id or location_id must be provided"}
    if note:
        data["note"] = note
    return await _post("/parameter/", data=data)


async def update_parameter(parameter_id: int, data_value: str | None = None,
                            note: str | None = None) -> dict:
    """Update an existing parameter value."""
    data: dict[str, Any] = {}
    if data_value is not None:
        data["data"] = data_value
    if note is not None:
        data["note"] = note
    if not data:
        return {"error": "No fields to update"}
    return await _patch(f"/parameter/{parameter_id}/", data=data)


async def delete_parameter(parameter_id: int) -> bool:
    """Delete a parameter value."""
    return await _delete(f"/parameter/{parameter_id}/")


# --- Image Upload ---

async def upload_part_image(part_id: int, image_bytes: bytes, filename: str = "photo.jpg", mime_type: str = "image/jpeg") -> dict:
    """Upload an image as the thumbnail for a Part.

    Uses multipart/form-data PATCH to set the Part's image field.
    """
    auth_header = {"Authorization": f"Token {settings.inventree_api_token}"}
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.patch(
            f"{API_BASE}/part/{part_id}/",
            headers=auth_header,
            files={"image": (filename, image_bytes, mime_type)},
        )
        resp.raise_for_status()
        return resp.json()


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


# --- Delete & Deactivate ---

async def deactivate_part(part_id: int) -> dict:
    """Deactivate (archive) a part by setting active=false."""
    return await _patch(f"/part/{part_id}/", data={"active": False})


async def delete_part(part_id: int) -> dict:
    """Delete a part. Deactivates it first if it is still active.

    Note: will fail if the part is locked or used in a BOM (unless the
    InvenTree setting PART_ALLOW_DELETE_FROM_ASSEMBLY is enabled).
    """
    # InvenTree requires the part to be inactive before deletion
    part = await _get(f"/part/{part_id}/")
    if part.get("active", True):
        await _patch(f"/part/{part_id}/", data={"active": False})
    await _delete(f"/part/{part_id}/")
    return {"deleted": True, "part_id": part_id, "name": part.get("name", "")}


async def delete_stock_item(stock_id: int) -> dict:
    """Delete a stock item."""
    # Fetch name/part info before deleting for a useful response
    item = await _get(f"/stock/{stock_id}/")
    await _delete(f"/stock/{stock_id}/")
    return {"deleted": True, "stock_id": stock_id, "part": item.get("part"), "quantity": item.get("quantity")}


async def delete_location(
    location_id: int,
    delete_stock_items: bool = False,
    delete_sub_locations: bool = False,
) -> dict:
    """Delete a stock location.

    By default, child locations and stock items are re-parented to the
    parent location (safe). Set flags to True to cascade-delete instead.
    """
    loc = await _get(f"/stock/location/{location_id}/")
    data = {}
    if delete_stock_items:
        data["delete_stock_items"] = True
    if delete_sub_locations:
        data["delete_sub_locations"] = True
    await _delete(f"/stock/location/{location_id}/", data=data or None)
    return {
        "deleted": True,
        "location_id": location_id,
        "name": loc.get("name", ""),
        "stock_items_deleted": delete_stock_items,
        "sub_locations_deleted": delete_sub_locations,
    }


async def delete_category(
    category_id: int,
    delete_parts: bool = False,
    delete_child_categories: bool = False,
) -> dict:
    """Delete a part category.

    By default, child categories and parts are re-parented to the parent
    category (safe). Set flags to True to cascade-delete instead.
    """
    cat = await _get(f"/part/category/{category_id}/")
    data = {}
    if delete_parts:
        data["delete_parts"] = True
    if delete_child_categories:
        data["delete_child_categories"] = True
    await _delete(f"/part/category/{category_id}/", data=data or None)
    return {
        "deleted": True,
        "category_id": category_id,
        "name": cat.get("name", ""),
        "parts_deleted": delete_parts,
        "child_categories_deleted": delete_child_categories,
    }
