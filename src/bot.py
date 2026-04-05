"""Telegram bot entrypoint — receives messages and photos, routes to the Gemini agent."""

import logging

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from config import settings
from agent import chat

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def _is_authorized(user_id: int) -> bool:
    if not settings.allowed_users:
        return True
    return user_id in settings.allowed_users


async def user_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    authorized = _is_authorized(uid)
    status = "ALLOWED" if authorized else "NOT ALLOWED"
    await update.message.reply_text(f"Your user ID: `{uid}`\nStatus: {status}", parse_mode="Markdown")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update.effective_user.id):
        return

    user_text = update.message.text or ""
    if not user_text.strip():
        return

    await update.message.chat.send_action("typing")

    try:
        reply = await chat(user_text)
        await update.message.reply_text(reply, parse_mode="Markdown")
    except Exception:
        logger.exception("Error processing message")
        await update.message.reply_text("Sorry, something went wrong. Please try again.")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update.effective_user.id):
        return

    caption = update.message.caption or "What is this item? Do I already have it in my inventory? Where should I put it?"

    await update.message.chat.send_action("typing")

    try:
        # Download the largest available photo
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        image_bytes = await file.download_as_bytearray()

        reply = await chat(caption, image_bytes=bytes(image_bytes), mime_type="image/jpeg")
        await update.message.reply_text(reply, parse_mode="Markdown")
    except Exception:
        logger.exception("Error processing photo")
        await update.message.reply_text("Sorry, I couldn't process that photo. Please try again.")


def main() -> None:
    app = Application.builder().token(settings.telegram_bot_token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("user_id", user_id))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot starting...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
