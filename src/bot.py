"""Telegram bot entrypoint — receives messages and photos, routes to the Gemini agent."""

import logging
from datetime import datetime, timezone

from telegram import BotCommand, Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from config import settings
from agent import chat, request_log

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


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

    await update.message.reply_text("\n".join(lines))


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update.effective_user.id):
        return

    user_text = update.message.text or ""
    if not user_text.strip():
        return

    await update.message.chat.send_action("typing")

    try:
        reply, _model = await chat(user_text)
        await update.message.reply_text(reply)
    except Exception:
        logger.exception("Error processing message")
        await update.message.reply_text("Sorry, something went wrong. Please try again.")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update.effective_user.id):
        return

    caption = update.message.caption or "What is this item? Do I already have it in my inventory? Where should I put it?"

    await update.message.chat.send_action("typing")

    try:
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        image_bytes = await file.download_as_bytearray()

        reply, _model = await chat(caption, image_bytes=bytes(image_bytes), mime_type="image/jpeg")
        await update.message.reply_text(reply)
    except Exception:
        logger.exception("Error processing photo")
        await update.message.reply_text("Sorry, I couldn't process that photo. Please try again.")


async def post_init(application: Application) -> None:
    """Register bot commands so they show in Telegram's command menu."""
    await application.bot.set_my_commands([
        BotCommand("start", "Show welcome message"),
        BotCommand("status", "Show API usage and model info"),
        BotCommand("user_id", "Show your Telegram user ID"),
    ])


def main() -> None:
    app = Application.builder().token(settings.telegram_bot_token).post_init(post_init).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("user_id", cmd_user_id))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot starting...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
