"""
Which rig a category hangs on.

garment_publish.py reads it for the rig, telegram_bot.py reads its keys for the
three buttons. It imports nothing, which is what lets both read it with no cycle.
"""

RIG_BY_CATEGORY = {"shirt": "top", "pants": "bottom", "skirt": "bottom"}
