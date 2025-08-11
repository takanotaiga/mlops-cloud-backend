from uuid import uuid4
import re

def get_uuid(uuid_lengh: int = 32) -> str:
    return re.sub(r'[^a-zA-Z0-9]', '', str(uuid4())).lower()[:uuid_lengh]
