"""Telegram bot entrypoint — receives messages and photos, routes to the Gemini agent."""

import asyncio
import logging
from datetime import datetime, timezone

from telegram import BotCommand, Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.error import BadRequest
from telegram.error import BadRequest

from config import settings
from agent import chat, request_log
from compaction import init_data_dir, refresh_context, context_refresh_loop
from session import clear_session, get_session, idle_compaction_loop

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
# Suppress httpx request logging — it includes the bot token in Telegram API URLs
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


async def _edit_with_html(msg, text: str) -> None:
    """Edit a message with HTML parsing, falling back to plain text."""
    try:
        await msg.edit_text(text, parse_mode="HTML", disable_web_page_preview=True)
    except BadRequest:
        logger.warning("HTML parse failed, falling back to plain text")
        await msg.edit_text(text)


def _is_authorized(user_id: int) -> bool:
    if not settings.allowed_users:
        return True
    return user_id in settings.allowed_users


async def cmd_user_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    authorized = _is_authorized(uid)
    status = "ALLOWED" if authorized else "NOT ALLOWED"
    await update.message.reply_text(f"Your user ID: {uid}\nStatus: {status}")


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update.effective_user.id):
        await update.message.reply_text("Unauthorized.")
        return
    await update.message.reply_text(
        "Hi! I'm your home inventory assistant.\n\n"
        "You can:\n"
        "- Ask me where something is\n"
        "- Send a photo and ask me to identify/add an item\n"
        "- Ask me to add, move, or count items\n"
        "- Ask 'where should I put this?'\n"
        "- Request an inventory summary\n\n"
        "Just type naturally or send a photo!"
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update.effective_user.id):
        return

    lines = ["Model usage today:"]
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if not request_log:
        lines.append("  No requests yet.")
    else:
        # Count requests per model for today
        counts: dict[str, int] = {}
        for entry in request_log:
            if entry["date"] == today:
                model = entry["model"]
                counts[model] = counts.get(model, 0) + 1

        if not counts:
            lines.append("  No requests today.")
        else:
            for model, count in sorted(counts.items()):
                lines.append(f"  {model}: {count}")

    lines.append(f"\nText chain: {', '.join(settings.text_models)}")
    lines.append(f"Vision chain: {', '.join(settings.vision_models)}")

    # Session info
    session = get_session(update.effective_user.id)
    lines.append(f"\nSession: {len(session.messages)} messages, ~{session.token_estimate} tokens")
    lines.append(f"Budget: {settings.context_budget} tokens")

    await update.message.reply_text("\n".join(lines))


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Clear the user's conversation history."""
    if not _is_authorized(update.effective_user.id):
        return
    clear_session(update.effective_user.id)
    await update.message.reply_text("Conversation history cleared.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update.effective_user.id):
        return

    user_text = update.message.text or ""
    if not user_text.strip():
        return

    # If the user replies to a message that has a photo, stash it on the session
    # so Gemini can attach it to a Part via upload_part_image.
    image_bytes = None
    replied = update.message.reply_to_message
    if replied and replied.photo:
        try:
            photo = replied.photo[-1]
            file = await context.bot.get_file(photo.file_id)
            raw = await file.download_as_bytearray()
            image_bytes = bytes(raw)
            session = get_session(update.effective_user.id)
            session.pending_image = image_bytes
            session.pending_image_mime = "image/jpeg"
            logger.info("Stashed image from replied-to message for user %s", update.effective_user.id)
        except Exception:
            logger.warning("Failed to fetch photo from replied-to message", exc_info=True)

    thinking_msg = await update.message.reply_text("Thinking...")

    try:
        reply, _model = await chat(update.effective_user.id, user_text)
        await _edit_with_html(thinking_msg, reply)
    except Exception:
        logger.exception("Error processing message")
        await thinking_msg.edit_text("Sorry, something went wrong. Please try again.")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update.effective_user.id):
        return

    caption = update.message.caption or "What is this item? Do I already have it in my inventory? Where should I put it?"

    thinking_msg = await update.message.reply_text("Thinking...")

    try:
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        image_bytes = bytes(await file.download_as_bytearray())

        # Stash the image on the session so upload_part_image can use it
        session = get_session(update.effective_user.id)
        session.pending_image = image_bytes
        session.pending_image_mime = "image/jpeg"

        reply, _model = await chat(update.effective_user.id, caption, image_bytes=image_bytes, mime_type="image/jpeg")
        await _edit_with_html(thinking_msg, reply)
    except Exception:
        logger.exception("Error processing photo")
        await thinking_msg.edit_text("Sorry, I couldn't process that photo. Please try again.")


async def post_init(application: Application) -> None:
    """Register bot commands, seed data dir, and start background tasks."""
    await application.bot.set_my_commands([
        BotCommand("start", "Show welcome message"),
        BotCommand("status", "Show API usage and session info"),
        BotCommand("clear", "Clear conversation history"),
        BotCommand("user_id", "Show your Telegram user ID"),
    ])
    # Seed data/ from sample-data/ if needed, then build initial context
    init_data_dir()
    await refresh_context()
    asyncio.create_task(context_refresh_loop())
    asyncio.create_task(idle_compaction_loop())


def main() -> None:
    app = Application.builder().token(settings.telegram_bot_token).post_init(post_init).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("user_id", cmd_user_id))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot starting...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
