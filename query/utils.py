from typing import Any, List, Optional


def first_result(payload: Any) -> Optional[Any]:
    """
    Extract the first logical row/value from SurrealDB's query response.
    Accepts envelope style: [{ status, time, result: [...] }] or raw lists.
    """
    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict) and "result" in first:
            results = first.get("result") or []
            return results[0] if results else None
        return first
    return None


def extract_results(payload: Any) -> List[Any]:
    """
    Extract the result list from SurrealDB's query response.
    Returns an empty list if none.
    """
    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict) and "result" in first:
            results = first.get("result") or []
            return list(results)
        # already a list of rows
        return list(payload)
    return []


def rid_leaf(rid: Any) -> str:
    """
    Convert a SurrealDB RecordID (e.g., 'table:id') to its leaf id string 'id'.
    Accepts raw strings or objects with a sensible __str__.
    """
    s = str(rid)
    return s.split(":", 1)[1] if ":" in s else s

