"""Gemini-powered agent that interprets user intent and calls InvenTree APIs.

Supports text and image inputs. Uses function-calling to interact with InvenTree.
Falls back through a configurable list of models on rate-limit (429) errors.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError

from config import settings
from compaction import get_context, get_prompt, refresh_context
from session import get_session
import inventree_client as inv

logger = logging.getLogger(__name__)

client = genai.Client(api_key=settings.gemini_api_key)

# In-memory request log for /status command (reset on restart)
request_log: list[dict] = []

# Functions that modify inventory state — context should be refreshed after these
WRITE_FUNCTIONS = {
    "create_part", "create_stock_item", "update_stock_quantity",
    "move_stock", "move_location", "create_location", "create_category",
    "deactivate_part", "delete_part", "delete_stock_item",
    "delete_location", "delete_category",
    "upload_part_image",
    "update_location", "update_part", "update_category", "move_category",
    "add_stock", "remove_stock", "transfer_stock",
    "create_parameter_template", "set_parameter", "update_parameter", "delete_parameter",
}

# Define the tools Gemini can call
TOOLS = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="search_parts",
                description="Search for parts/items in the inventory by name, description, or keywords",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "query": types.Schema(type=types.Type.STRING, description="Search query"),
                        "limit": types.Schema(type=types.Type.INTEGER, description="Max results (default 20)"),
                    },
                    required=["query"],
                ),
            ),
            types.FunctionDeclaration(
                name="get_part",
                description="Get details of a specific part by its ID",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "part_id": types.Schema(type=types.Type.INTEGER, description="Part ID"),
                    },
                    required=["part_id"],
                ),
            ),
            types.FunctionDeclaration(
                name="list_stock",
                description="List stock items. Can filter by part ID or location ID to see what's at a location or how much stock a part has.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "part_id": types.Schema(type=types.Type.INTEGER, description="Filter by part ID"),
                        "location_id": types.Schema(type=types.Type.INTEGER, description="Filter by location ID"),
                        "limit": types.Schema(type=types.Type.INTEGER, description="Max results (default 50)"),
                    },
                ),
            ),
            types.FunctionDeclaration(
                name="list_locations",
                description="List all stock locations (rooms, shelves, drawers, boxes). Optionally filter by parent location.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "parent": types.Schema(type=types.Type.INTEGER, description="Parent location ID to list children of"),
                    },
                ),
            ),
            types.FunctionDeclaration(
                name="list_categories",
                description="List all part categories (Electronics, Tools, Kitchen, etc.). Optionally filter by parent category.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "parent": types.Schema(type=types.Type.INTEGER, description="Parent category ID to list children of"),
                    },
                ),
            ),
            types.FunctionDeclaration(
                name="create_part",
                description="Create a new part/item in the inventory",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "name": types.Schema(type=types.Type.STRING, description="Part name"),
                        "description": types.Schema(type=types.Type.STRING, description="Part description"),
                        "category_id": types.Schema(type=types.Type.INTEGER, description="Category ID"),
                    },
                    required=["name", "description", "category_id"],
                ),
            ),
            types.FunctionDeclaration(
                name="create_stock_item",
                description="Add stock for a part at a specific location (e.g. put 3 of item X on shelf Y)",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "part_id": types.Schema(type=types.Type.INTEGER, description="Part ID"),
                        "quantity": types.Schema(type=types.Type.NUMBER, description="Quantity"),
                        "location_id": types.Schema(type=types.Type.INTEGER, description="Location ID"),
                    },
                    required=["part_id", "quantity", "location_id"],
                ),
            ),
            types.FunctionDeclaration(
                name="update_stock_quantity",
                description="Update the quantity of an existing stock item",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "stock_id": types.Schema(type=types.Type.INTEGER, description="Stock item ID"),
                        "quantity": types.Schema(type=types.Type.NUMBER, description="New quantity"),
                    },
                    required=["stock_id", "quantity"],
                ),
            ),
            types.FunctionDeclaration(
                name="move_stock",
                description="Move a stock item to a different location",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "stock_id": types.Schema(type=types.Type.INTEGER, description="Stock item ID"),
                        "location_id": types.Schema(type=types.Type.INTEGER, description="New location ID"),
                    },
                    required=["stock_id", "location_id"],
                ),
            ),
            types.FunctionDeclaration(
                name="create_location",
                description="Create a new stock location (room, shelf, drawer, box)",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "name": types.Schema(type=types.Type.STRING, description="Location name"),
                        "description": types.Schema(type=types.Type.STRING, description="Location description"),
                        "parent": types.Schema(type=types.Type.INTEGER, description="Parent location ID"),
                    },
                    required=["name"],
                ),
            ),
            types.FunctionDeclaration(
                name="move_location",
                description="Move a location to a new parent (re-parent). All child locations and stock items inside it move automatically. Use this instead of creating a new location when relocating furniture or portable containers.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "location_id": types.Schema(type=types.Type.INTEGER, description="Location ID to move"),
                        "parent": types.Schema(type=types.Type.INTEGER, description="New parent location ID"),
                    },
                    required=["location_id", "parent"],
                ),
            ),
            types.FunctionDeclaration(
                name="create_category",
                description="Create a new part category",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "name": types.Schema(type=types.Type.STRING, description="Category name"),
                        "description": types.Schema(type=types.Type.STRING, description="Category description"),
                        "parent": types.Schema(type=types.Type.INTEGER, description="Parent category ID"),
                    },
                    required=["name"],
                ),
            ),
            types.FunctionDeclaration(
                name="get_inventory_summary",
                description="Get a high-level summary of the entire inventory (total parts, stock items, locations, categories)",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={},
                ),
            ),
            types.FunctionDeclaration(
                name="deactivate_part",
                description="Deactivate (archive) a part without deleting it. The part remains in the database but is marked inactive.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "part_id": types.Schema(type=types.Type.INTEGER, description="Part ID to deactivate"),
                    },
                    required=["part_id"],
                ),
            ),
            types.FunctionDeclaration(
                name="delete_part",
                description="Permanently delete a part. Automatically deactivates it first if needed. Fails if the part is locked or used in a BOM.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "part_id": types.Schema(type=types.Type.INTEGER, description="Part ID to delete"),
                    },
                    required=["part_id"],
                ),
            ),
            types.FunctionDeclaration(
                name="delete_stock_item",
                description="Permanently delete a stock item (remove a physical stock entry)",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "stock_id": types.Schema(type=types.Type.INTEGER, description="Stock item ID to delete"),
                    },
                    required=["stock_id"],
                ),
            ),
            types.FunctionDeclaration(
                name="delete_location",
                description="Delete a stock location. By default, child locations and stock items are safely re-parented to the parent. Set flags to cascade-delete instead.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "location_id": types.Schema(type=types.Type.INTEGER, description="Location ID to delete"),
                        "delete_stock_items": types.Schema(type=types.Type.BOOLEAN, description="Also delete all stock items in this location (default: false, re-parents them)"),
                        "delete_sub_locations": types.Schema(type=types.Type.BOOLEAN, description="Also delete all child locations (default: false, re-parents them)"),
                    },
                    required=["location_id"],
                ),
            ),
            types.FunctionDeclaration(
                name="delete_category",
                description="Delete a part category. By default, child categories and parts are safely re-parented to the parent. Set flags to cascade-delete instead.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "category_id": types.Schema(type=types.Type.INTEGER, description="Category ID to delete"),
                        "delete_parts": types.Schema(type=types.Type.BOOLEAN, description="Also delete all parts in this category (default: false, re-parents them)"),
                        "delete_child_categories": types.Schema(type=types.Type.BOOLEAN, description="Also delete all child categories (default: false, re-parents them)"),
                    },
                    required=["category_id"],
                ),
            ),
            types.FunctionDeclaration(
                name="upload_part_image",
                description=(
                    "Upload the user's photo as the thumbnail image for a Part. "
                    "Uses the most recent photo from this conversation (sent directly "
                    "or from a replied-to message). Only call this when the user's photo "
                    "depicts the item the Part represents. Do NOT call this if the photo "
                    "is of something else (e.g. a shelf, a room, or an unrelated item)."
                ),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "part_id": types.Schema(type=types.Type.INTEGER, description="Part ID to attach the image to"),
                    },
                    required=["part_id"],
                ),
            ),
            types.FunctionDeclaration(
                name="get_stock_item",
                description="Get details of a specific stock item by its ID",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "stock_id": types.Schema(type=types.Type.INTEGER, description="Stock item ID"),
                    },
                    required=["stock_id"],
                ),
            ),
            types.FunctionDeclaration(
                name="get_category",
                description="Get details of a specific part category by its ID",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "category_id": types.Schema(type=types.Type.INTEGER, description="Category ID"),
                    },
                    required=["category_id"],
                ),
            ),
            types.FunctionDeclaration(
                name="get_location",
                description="Get details of a specific stock location by its ID",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "location_id": types.Schema(type=types.Type.INTEGER, description="Location ID"),
                    },
                    required=["location_id"],
                ),
            ),
            types.FunctionDeclaration(
                name="update_location",
                description="Update a stock location's name and/or description (rename a room, shelf, etc.)",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "location_id": types.Schema(type=types.Type.INTEGER, description="Location ID to update"),
                        "name": types.Schema(type=types.Type.STRING, description="New name for the location"),
                        "description": types.Schema(type=types.Type.STRING, description="New description for the location"),
                    },
                    required=["location_id"],
                ),
            ),
            types.FunctionDeclaration(
                name="update_part",
                description="Update a part's name, description, category, or keywords",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "part_id": types.Schema(type=types.Type.INTEGER, description="Part ID to update"),
                        "name": types.Schema(type=types.Type.STRING, description="New name"),
                        "description": types.Schema(type=types.Type.STRING, description="New description"),
                        "category_id": types.Schema(type=types.Type.INTEGER, description="New category ID"),
                        "keywords": types.Schema(type=types.Type.STRING, description="New keywords (comma-separated)"),
                    },
                    required=["part_id"],
                ),
            ),
            types.FunctionDeclaration(
                name="update_category",
                description="Update a part category's name and/or description",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "category_id": types.Schema(type=types.Type.INTEGER, description="Category ID to update"),
                        "name": types.Schema(type=types.Type.STRING, description="New name"),
                        "description": types.Schema(type=types.Type.STRING, description="New description"),
                    },
                    required=["category_id"],
                ),
            ),
            types.FunctionDeclaration(
                name="move_category",
                description="Move a part category to a new parent (re-parent). All child categories and parts stay nested under it.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "category_id": types.Schema(type=types.Type.INTEGER, description="Category ID to move"),
                        "parent": types.Schema(type=types.Type.INTEGER, description="New parent category ID"),
                    },
                    required=["category_id", "parent"],
                ),
            ),
            types.FunctionDeclaration(
                name="add_stock",
                description="Add quantity to an existing stock item with proper tracking history. Use this when the user receives more of an item or corrects a count upward.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "stock_id": types.Schema(type=types.Type.INTEGER, description="Stock item ID"),
                        "quantity": types.Schema(type=types.Type.NUMBER, description="Quantity to add"),
                        "notes": types.Schema(type=types.Type.STRING, description="Reason for the adjustment"),
                    },
                    required=["stock_id", "quantity"],
                ),
            ),
            types.FunctionDeclaration(
                name="remove_stock",
                description="Remove quantity from an existing stock item with proper tracking history. Use this when items are consumed, used up, or the count needs correcting downward.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "stock_id": types.Schema(type=types.Type.INTEGER, description="Stock item ID"),
                        "quantity": types.Schema(type=types.Type.NUMBER, description="Quantity to remove"),
                        "notes": types.Schema(type=types.Type.STRING, description="Reason for the removal"),
                    },
                    required=["stock_id", "quantity"],
                ),
            ),
            types.FunctionDeclaration(
                name="transfer_stock",
                description="Transfer a quantity of stock from its current location to a different location. If transferring less than the full quantity, InvenTree splits the stock item automatically. Creates tracking history.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "stock_id": types.Schema(type=types.Type.INTEGER, description="Stock item ID to transfer from"),
                        "quantity": types.Schema(type=types.Type.NUMBER, description="Quantity to transfer"),
                        "location_id": types.Schema(type=types.Type.INTEGER, description="Destination location ID"),
                        "notes": types.Schema(type=types.Type.STRING, description="Reason for the transfer"),
                    },
                    required=["stock_id", "quantity", "location_id"],
                ),
            ),
            types.FunctionDeclaration(
                name="list_parameter_templates",
                description="List all available parameter templates — the types of attributes that can be tracked on parts or locations (e.g. Weight, Dimensions, Purchase Price, Warranty Expiry).",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "limit": types.Schema(type=types.Type.INTEGER, description="Max results (default 100)"),
                    },
                ),
            ),
            types.FunctionDeclaration(
                name="create_parameter_template",
                description="Create a new parameter template — define a new type of attribute that can be tracked (e.g. 'Purchase Price' with units 'EUR', or 'Warranty Expiry' with no units).",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "name": types.Schema(type=types.Type.STRING, description="Parameter name (e.g. 'Weight', 'Purchase Price')"),
                        "units": types.Schema(type=types.Type.STRING, description="Physical units (e.g. 'kg', 'EUR', 'mm'). Empty for unitless."),
                        "description": types.Schema(type=types.Type.STRING, description="Human-readable description"),
                        "checkbox": types.Schema(type=types.Type.BOOLEAN, description="If true, parameter is a boolean checkbox"),
                        "choices": types.Schema(type=types.Type.STRING, description="Comma-separated valid choices (e.g. 'Small,Medium,Large')"),
                    },
                    required=["name"],
                ),
            ),
            types.FunctionDeclaration(
                name="list_parameters",
                description="List parameter values (specs/attributes) for a specific part or location. Returns things like weight, dimensions, purchase price, etc.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "part_id": types.Schema(type=types.Type.INTEGER, description="Part ID to get parameters for"),
                        "location_id": types.Schema(type=types.Type.INTEGER, description="Location ID to get parameters for (use part_id OR location_id, not both)"),
                        "template_id": types.Schema(type=types.Type.INTEGER, description="Filter by parameter template ID"),
                        "limit": types.Schema(type=types.Type.INTEGER, description="Max results (default 100)"),
                    },
                ),
            ),
            types.FunctionDeclaration(
                name="set_parameter",
                description="Set a parameter value on a part or location. Requires a template_id (use list_parameter_templates to find or create_parameter_template to make one). The value is always a string.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "template_id": types.Schema(type=types.Type.INTEGER, description="Parameter template ID (what attribute)"),
                        "data_value": types.Schema(type=types.Type.STRING, description="The value to set (as string, e.g. '2.5', '2025-12-31', 'Large')"),
                        "part_id": types.Schema(type=types.Type.INTEGER, description="Part ID to set parameter on (use part_id OR location_id)"),
                        "location_id": types.Schema(type=types.Type.INTEGER, description="Location ID to set parameter on"),
                        "note": types.Schema(type=types.Type.STRING, description="Optional note about this value"),
                    },
                    required=["template_id", "data_value"],
                ),
            ),
            types.FunctionDeclaration(
                name="update_parameter",
                description="Update an existing parameter value by its parameter ID (not template ID). Use list_parameters to find the parameter ID first.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "parameter_id": types.Schema(type=types.Type.INTEGER, description="Parameter instance ID to update"),
                        "data_value": types.Schema(type=types.Type.STRING, description="New value"),
                        "note": types.Schema(type=types.Type.STRING, description="Updated note"),
                    },
                    required=["parameter_id"],
                ),
            ),
            types.FunctionDeclaration(
                name="delete_parameter",
                description="Delete a parameter value from a part or location by its parameter ID.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "parameter_id": types.Schema(type=types.Type.INTEGER, description="Parameter instance ID to delete"),
                    },
                    required=["parameter_id"],
                ),
            ),
        ]
    )
]

def _build_system_prompt(user_id: int) -> str:
    """Assemble system prompt from data/prompt.txt + inventory context + per-user summary."""
    prompt = get_prompt()
    context = get_context()
    parts = [prompt, f"\n--- INVENTORY CONTEXT ---\n{context}"]

    session = get_session(user_id)
    if session.pending_image is not None:
        parts.append(
            "\n--- PHOTO AVAILABLE ---\n"
            "A photo from this conversation is available for upload. "
            "If you create or update a Part that matches this photo, "
            "call upload_part_image(part_id) to attach it as the Part's image."
        )
    if session.summary:
        parts.append(f"\n--- CONVERSATION CONTEXT ---\n{session.summary}")

    return "\n".join(parts)

# Map function names to actual callables
FUNCTION_MAP: dict[str, Any] = {
    "search_parts": inv.search_parts,
    "get_part": inv.get_part,
    "get_stock_item": inv.get_stock_item,
    "get_category": inv.get_category,
    "get_location": inv.get_location,
    "list_stock": inv.list_stock,
    "list_locations": inv.list_locations,
    "list_categories": inv.list_categories,
    "create_part": inv.create_part,
    "create_stock_item": inv.create_stock_item,
    "update_stock_quantity": inv.update_stock_quantity,
    "move_stock": inv.move_stock,
    "move_location": inv.move_location,
    "update_location": inv.update_location,
    "update_part": inv.update_part,
    "update_category": inv.update_category,
    "move_category": inv.move_category,
    "add_stock": inv.add_stock,
    "remove_stock": inv.remove_stock,
    "transfer_stock": inv.transfer_stock,
    "create_location": inv.create_location,
    "create_category": inv.create_category,
    "get_inventory_summary": inv.get_inventory_summary,
    "deactivate_part": inv.deactivate_part,
    "delete_part": inv.delete_part,
    "delete_stock_item": inv.delete_stock_item,
    "delete_location": inv.delete_location,
    "delete_category": inv.delete_category,
    "list_parameter_templates": inv.list_parameter_templates,
    "create_parameter_template": inv.create_parameter_template,
    "list_parameters": inv.list_parameters,
    "set_parameter": inv.set_parameter,
    "update_parameter": inv.update_parameter,
    "delete_parameter": inv.delete_parameter,
}


async def _execute_function_call(fn_call: types.FunctionCall, user_id: int) -> Any:
    """Execute a function call from Gemini and return the result.

    After write operations, refreshes the inventory context snapshot.
    upload_part_image is handled specially: it pulls the pending image from
    the user's session rather than receiving bytes via function args.
    """
    fn_name = fn_call.name
    fn_args = dict(fn_call.args) if fn_call.args else {}
    logger.info("Calling %s(%s)", fn_name, fn_args)

    # Special handling for upload_part_image — inject pending image from session
    if fn_name == "upload_part_image":
        session = get_session(user_id)
        if session.pending_image is None:
            return {"error": "No photo available. The user needs to send or reply to a photo first."}
        part_id = int(fn_args["part_id"])
        try:
            result = await inv.upload_part_image(
                part_id=part_id,
                image_bytes=session.pending_image,
                mime_type=session.pending_image_mime,
            )
            session.pending_image = None  # consumed
            await refresh_context()
            return result
        except Exception as e:
            logger.exception("Error uploading image for part %s", part_id)
            return {"error": str(e)}

    fn = FUNCTION_MAP.get(fn_name)
    if fn is None:
        return {"error": f"Unknown function: {fn_name}"}

    try:
        result = await fn(**fn_args)
        if fn_name in WRITE_FUNCTIONS:
            await refresh_context()
        return result
    except Exception as e:
        logger.exception("Error calling %s", fn_name)
        return {"error": str(e)}


async def _generate_with_fallback(
    models: list[str],
    contents: list[types.Content],
    user_id: int,
    preferred_model: str | None = None,
) -> tuple[types.GenerateContentResponse, str]:
    """Try each model in the fallback list. On 429/503, wait briefly then try next.
    If preferred_model is set (mid-conversation), try it first with a retry.
    Returns (response, model_name)."""
    # If we're mid-conversation, prioritize the model we started with
    order = list(models)
    if preferred_model and preferred_model in order:
        order.remove(preferred_model)
        order.insert(0, preferred_model)

    last_error = None
    for model in order:
        # Try up to 2 attempts per model (with a wait on 429/503)
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=_build_system_prompt(user_id),
                        tools=TOOLS,
                        temperature=0.3,
                    ),
                )
                logger.info("Success with model: %s", model)
                request_log.append({
                    "model": model,
                    "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                })
                return response, model
            except ClientError as e:
                if e.code == 429:
                    last_error = e
                    if attempt == 0:
                        logger.warning("Rate limited on %s, waiting 15s before retry...", model)
                        await asyncio.sleep(15)
                    else:
                        logger.warning("Rate limited on %s again, trying next model...", model)
                    continue
                raise
            except ServerError as e:
                last_error = e
                if attempt == 0:
                    logger.warning("Server error (%s) on %s, waiting 10s before retry...", e.code, model)
                    await asyncio.sleep(10)
                else:
                    logger.warning("Server error (%s) on %s again, trying next model...", e.code, model)
                continue
    raise last_error


async def chat(user_id: int, user_message: str, image_bytes: bytes | None = None, mime_type: str = "image/jpeg") -> tuple[str, str]:
    """Process a user message (with optional image) and return (reply_text, model_used).

    Maintains per-user conversation history. On each call:
    1. Append the user message to the session
    2. Check if hot compaction is needed (90% of budget)
    3. Send full history to Gemini with function-calling loop
    4. Append all model/function-result messages to the session
    """
    session = get_session(user_id)

    # Build the user content parts
    parts: list[types.Part] = []
    if image_bytes:
        parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
    parts.append(types.Part.from_text(text=user_message))

    user_content = types.Content(role="user", parts=parts)
    session.add_message(user_content)

    # Hot compaction: if we're at 90% budget, compact before sending
    if session.needs_hot_compaction():
        logger.info(
            "Hot compaction for user %s (%d tokens, budget %d)",
            user_id, session.token_estimate, settings.context_budget,
        )
        await session.compact(keep_raw=settings.hot_compaction_keep_raw)

    models = settings.vision_models if image_bytes else settings.text_models
    logger.info("Model fallback chain: %s", models)

    used_model = models[0]

    # Loop: send to Gemini, execute any function calls, feed results back
    max_rounds = 5
    for round_num in range(max_rounds):
        try:
            response, used_model = await _generate_with_fallback(
                models, session.messages, user_id,
                preferred_model=used_model if round_num > 0 else None,
            )
        except ClientError as e:
            # INVALID_ARGUMENT with "function response turn" means the persisted
            # session has an orphaned function-call message (no matching response),
            # which happens when the bot crashes mid-request. Clear history and retry.
            if e.code == 400 and "function response turn" in str(e).lower():
                logger.warning(
                    "Session %s has corrupt history (orphaned function call). "
                    "Clearing and retrying.", user_id
                )
                session.clear()
                session.add_message(user_content)
                continue
            raise

        candidate = response.candidates[0]
        content = candidate.content

        # Check if the model wants to call functions
        fn_calls = [part.function_call for part in content.parts if part.function_call]

        if not fn_calls:
            # Final text response — add to session and return
            session.add_message(content)
            session.save_to_disk()
            text_parts = [part.text for part in content.parts if part.text]
            reply = "\n".join(text_parts) if text_parts else "I couldn't process that request."
            return reply, used_model

        # Execute all function calls
        session.add_message(content)

        fn_response_parts: list[types.Part] = []
        for fn_call in fn_calls:
            result = await _execute_function_call(fn_call, user_id)
            fn_response_parts.append(
                types.Part.from_function_response(
                    name=fn_call.name,
                    response={"result": result},
                )
            )

        fn_results_content = types.Content(role="user", parts=fn_response_parts)
        session.add_message(fn_results_content)

    session.save_to_disk()
    return "I ran into a loop trying to process your request. Please try rephrasing.", used_model
