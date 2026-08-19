"""
Which rig a category hangs on.

garment_publish.py reads it for the rig, anchor_server.py reads its keys for
/categories. It imports nothing, which is what lets both read it with no cycle.
"""

RIG_BY_CATEGORY = {"shirt": "top", "pants": "bottom", "skirt": "bottom"}
