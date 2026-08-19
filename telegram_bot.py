"""
Telegram in, garment on the mirror.

Shaped like parser_thread.py: available() then start(), the thread owned here,
so main.py never sees an event loop. No token on a dev machine means no bot,
the same way no GPU means no parser - not a crash, not a branch in main.py.

The anchor page is offered the same way: with no ANCHOR_APP_URL the bot runs
and simply never shows the button.
"""
import asyncio
import os
import threading
from pathlib import Path

from dotenv import load_dotenv
from telegram import KeyboardButton, ReplyKeyboardMarkup, WebAppInfo
from telegram.error import TelegramError
from telegram.ext import Application, CommandHandler, MessageHandler, filters

import anchor_server
import anchor_session
import config
import garment_publish
import normalize
from garment_types import RIG_BY_CATEGORY

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
PASSWORD = os.getenv("GARMENT_BOT_PASSWORD")

# The categories are the buttons, so neither list can drift from the other.
KEYBOARD = ReplyKeyboardMarkup([list(RIG_BY_CATEGORY)], resize_keyboard=True, one_time_keyboard=True)

# A button's text is what comes back as a message, so the two live in one place.
PUBLISH_AS_IS = "Publish as is"
PLACE_POINTS = "Place points"

# Per chat, not one global flag: the first person through must not unlock the
# bot for everyone. It lives for the process, so a restart locks every chat.
_unlocked = set()

_labels = None   # parser class names, or None on a machine with no GPU


def available():
    """No token or no password means the bot simply does not run."""
    return bool(TOKEN and PASSWORD)


def start(labels):
    """Wipe the bot folder, then poll Telegram on a daemon thread."""
    global _labels
    _labels = labels
    _wipe()
    # daemon for parser_thread.py's reason: `q` has to end the process even
    # mid-upload, and a half-finished garment is nothing worth protecting.
    threading.Thread(target=_run, daemon=True).start()


def _wipe():
    """Nothing an upload produced outlives a restart. At startup rather than at
    shutdown, because a crash or a power cut never reaches a shutdown handler."""
    directory = Path(config.BOT_GARMENT_DIR)
    for pattern in ("*.png", "*.anchors.json"):
        for f in directory.glob(pattern):
            f.unlink()


def _run():
    """The bot's own event loop, on its own thread: python-telegram-bot is
    asyncio, and the render loop is a synchronous cv2 loop that owns the window."""
    asyncio.set_event_loop(asyncio.new_event_loop())

    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", _on_start))
    # Before the text handler: what the page sends is a message like any other,
    # and only its update type tells it apart from someone typing.
    app.add_handler(MessageHandler(filters.StatusUpdate.WEB_APP_DATA, _on_anchors))
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.ALL, _on_image))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _on_text))
    app.run_polling(stop_signals=None)   # signal handlers only install on the main thread


async def _on_start(update, context):
    await update.message.reply_text("Send the password.")


async def _on_text(update, context):
    text = update.message.text.strip()
    if update.effective_chat.id not in _unlocked:
        await _unlock(update, text)
    elif text in RIG_BY_CATEGORY:
        await _offer(update, text)
    elif text == PUBLISH_AS_IS:
        await _publish_draft(update)
    else:
        await update.message.reply_text("Send a photo of a garment.")


async def _unlock(update, text):
    if text != PASSWORD:
        await update.message.reply_text("Wrong password.")
        return

    _unlocked.add(update.effective_chat.id)
    # Best-effort only: Telegram will not let a bot delete a user's message in
    # a private chat, so the reply has to ask as well.
    try:
        await update.message.delete()
    except TelegramError:
        pass
    await update.message.reply_text("Unlocked - delete that password message. Now send a photo of a garment.")


async def _on_image(update, context):
    """Photo or document in, normalized PNG bytes parked until a button press."""
    message = update.message
    if update.effective_chat.id not in _unlocked:
        await message.reply_text("Send the password first.")
        return

    if message.photo:
        # A note, not a rejection: the input is capped to 1600 px anyway and
        # rembg masks at 320x320, so Telegram's re-encode costs little.
        await message.reply_text("Tip: sending as a file keeps more detail.")
        file_id = message.photo[-1].file_id   # the largest size Telegram kept
    else:
        file_id = message.document.file_id

    try:
        telegram_file = await context.bot.get_file(file_id)
        data = await telegram_file.download_as_bytearray()
    except TelegramError as e:
        await message.reply_text(f"Telegram would not hand that file over ({e}). Its cap is 20 MB.")
        return

    # Broad on purpose: these are arbitrary bytes off a phone, and an unhandled
    # error here would be logged and never answered - the user just waits.
    try:
        anchor_session.park_photo(update.effective_chat.id, normalize.to_png(bytes(data)))
    except Exception as e:
        await message.reply_text(f"Couldn't read that as an image ({e}).")
        return

    await message.reply_text("What is it?", reply_markup=KEYBOARD)


async def _offer(update, category):
    """A category press: cut the garment out, draft its anchors, show them, and
    offer whichever of the two ways on there is."""
    chat_id = update.effective_chat.id
    png = anchor_session.take_photo(chat_id)
    if png is None:
        await update.message.reply_text("Send a photo first.")
        return

    await update.message.reply_text(f"Working on the {category}...")
    # Seconds of numpy, run straight on this thread: it blocks the bot's own
    # loop and nothing else, since the render loop is a different thread.
    try:
        cutout, sidecar, trusted = garment_publish.prepare(png, category)
        preview = anchor_session.render_preview(anchor_session.decode(cutout), sidecar)
        anchor_session.new_session(chat_id, cutout, sidecar, category)
    except Exception as e:
        await update.message.reply_text(f"That one failed to build: {e}")
        return

    buttons = []
    # Only a measured top is worth publishing unseen; everything else is a
    # spread over the bounding box, and nobody wants that on the mirror.
    if trusted:
        buttons.append(KeyboardButton(PUBLISH_AS_IS))
    if anchor_server.APP_URL:
        # No token in the URL any more: the page asks for the password and
        # then opens on the one session, which is the one just parked here.
        buttons.append(KeyboardButton(PLACE_POINTS, web_app=WebAppInfo(url=anchor_server.APP_URL)))
    if not buttons:
        anchor_session.end_session(chat_id)
        if RIG_BY_CATEGORY[category] == "top":
            await update.message.reply_text(
                "couldn't find the shoulders - try a flatter photo against a plain background")
        else:
            await update.message.reply_text(f"{category} needs the anchor page, and ANCHOR_APP_URL is not set")
        return

    await update.message.reply_photo(
        preview, caption="Here is where the points landed.",
        reply_markup=ReplyKeyboardMarkup([buttons], resize_keyboard=True, one_time_keyboard=True))


async def _publish_draft(update):
    """Publish as is: the draft, exactly as the preview showed it."""
    session = anchor_session.get_session()
    if session is None:
        await update.message.reply_text("Send a photo first.")
        return
    await _build(update, session, session["sidecar"])


async def _on_anchors(update, context):
    """Save on the anchor page: Telegram delivers its JSON as an ordinary message."""
    session = anchor_session.get_session()
    if session is None:
        await update.message.reply_text("Send a photo first.")
        return

    try:
        sidecar = anchor_session.validate(update.message.web_app_data.data, session["sidecar"],
                                          session["category"], session["width"], session["height"])
    except ValueError as e:
        # The session outlives a bad save, so the same button opens the page again.
        await update.message.reply_text(str(e))
        return
    await _build(update, session, sidecar)


async def _build(update, session, sidecar):
    """The half of the pipeline after the anchors, wherever they came from."""
    try:
        reply = garment_publish.finish(session["cutout"], sidecar, session["category"], _labels)
    except Exception as e:
        reply = f"That one failed to build: {e}"
    anchor_session.end_session(update.effective_chat.id)
    await update.message.reply_text(reply)
