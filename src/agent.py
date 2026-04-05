"""Gemini-powered agent that interprets user intent and calls InvenTree APIs.

Supports text and image inputs. Uses function-calling to interact with InvenTree.
Falls back through a configurable list of models on rate-limit (429) errors.
"""

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from google.genai.errors import ClientError

from config import settings
from compaction import get_context
import inventree_client as inv

logger = logging.getLogger(__name__)

client = genai.Client(api_key=settings.gemini_api_key)

# In-memory request log for /status command (reset on restart)
request_log: list[dict] = []

PROMPT_FILE = Path(__file__).parent / "prompt.txt"

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
        ]
    )
]

def _build_system_prompt() -> str:
    """Assemble system prompt from prompt.txt + live context snapshot."""
    prompt = PROMPT_FILE.read_text(encoding="utf-8")
    context = get_context()
    return f"{prompt}\n\n--- INVENTORY CONTEXT ---\n{context}"

# Map function names to actual callables
FUNCTION_MAP: dict[str, Any] = {
    "search_parts": inv.search_parts,
    "get_part": inv.get_part,
    "list_stock": inv.list_stock,
    "list_locations": inv.list_locations,
    "list_categories": inv.list_categories,
    "create_part": inv.create_part,
    "create_stock_item": inv.create_stock_item,
    "update_stock_quantity": inv.update_stock_quantity,
    "move_stock": inv.move_stock,
    "create_location": inv.create_location,
    "create_category": inv.create_category,
    "get_inventory_summary": inv.get_inventory_summary,
}


async def _execute_function_call(fn_call: types.FunctionCall) -> Any:
    """Execute a function call from Gemini and return the result."""
    fn_name = fn_call.name
    fn_args = dict(fn_call.args) if fn_call.args else {}
    logger.info("Calling %s(%s)", fn_name, fn_args)

    fn = FUNCTION_MAP.get(fn_name)
    if fn is None:
        return {"error": f"Unknown function: {fn_name}"}

    try:
        result = await fn(**fn_args)
        return result
    except Exception as e:
        logger.exception("Error calling %s", fn_name)
        return {"error": str(e)}


async def _generate_with_fallback(
    models: list[str],
    contents: list[types.Content],
    preferred_model: str | None = None,
) -> tuple[types.GenerateContentResponse, str]:
    """Try each model in the fallback list. On 429, wait briefly then try next.
    If preferred_model is set (mid-conversation), try it first with a retry.
    Returns (response, model_name)."""
    # If we're mid-conversation, prioritize the model we started with
    order = list(models)
    if preferred_model and preferred_model in order:
        order.remove(preferred_model)
        order.insert(0, preferred_model)

    last_error = None
    for model in order:
        # Try up to 2 attempts per model (with a wait on 429)
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=_build_system_prompt(),
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
                if e.status_code == 429:
                    last_error = e
                    if attempt == 0:
                        logger.warning("Rate limited on %s, waiting 15s before retry...", model)
                        await asyncio.sleep(15)
                    else:
                        logger.warning("Rate limited on %s again, trying next model...", model)
                    continue
                raise
    raise last_error


async def chat(user_message: str, image_bytes: bytes | None = None, mime_type: str = "image/jpeg") -> tuple[str, str]:
    """Process a user message (with optional image) and return (reply_text, model_used).

    This handles multi-turn function calling: Gemini may call one or more functions,
    and we loop until it produces a final text response. On 429, waits and retries
    or falls back to the next model in the configured list.
    """
    # Build the user content parts
    parts: list[types.Part] = []
    if image_bytes:
        parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
    parts.append(types.Part.from_text(text=user_message))

    contents: list[types.Content] = [
        types.Content(role="user", parts=parts),
    ]

    models = settings.vision_models if image_bytes else settings.text_models
    logger.info("Model fallback chain: %s", models)

    used_model = models[0]

    # Loop: send to Gemini, execute any function calls, feed results back
    max_rounds = 5
    for round_num in range(max_rounds):
        response, used_model = await _generate_with_fallback(
            models, contents, preferred_model=used_model if round_num > 0 else None
        )

        candidate = response.candidates[0]
        content = candidate.content

        # Check if the model wants to call functions
        fn_calls = [part.function_call for part in content.parts if part.function_call]

        if not fn_calls:
            # Final text response
            text_parts = [part.text for part in content.parts if part.text]
            reply = "\n".join(text_parts) if text_parts else "I couldn't process that request."
            return reply, used_model

        # Execute all function calls
        contents.append(content)  # add the model's response with function calls

        fn_response_parts: list[types.Part] = []
        for fn_call in fn_calls:
            result = await _execute_function_call(fn_call)
            fn_response_parts.append(
                types.Part.from_function_response(
                    name=fn_call.name,
                    response={"result": result},
                )
            )

        contents.append(types.Content(role="user", parts=fn_response_parts))

    return "I ran into a loop trying to process your request. Please try rephrasing.", used_model
