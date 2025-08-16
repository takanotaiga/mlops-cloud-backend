from uuid import uuid4
import re

def get_uuid(uuid_lengh: int = 32) -> str:
    """Return a lowercase alphanumeric UUID string cropped to length.

    Note: keeps compatibility with existing call sites (param name typo preserved).
    """
    return re.sub(r"[^a-zA-Z0-9]", "", str(uuid4())).lower()[:uuid_lengh]


def add_uuid_prefix(name: str, uuid_length: int = 8, sep: str = "_") -> str:
    """Prefix a name (e.g., filename) with a short UUID.

    Examples:
        add_uuid_prefix("group_001.mp4") -> "a1b2c3d4_group_001.mp4"
    """
    return f"{get_uuid(uuid_length)}{sep}{name}"
