"""Lossless identity keys shared by repaired works, graphs and browser exports.

OpenAlex author IDs stay A<key>. Mapademic-owned identities use M<key>, with
positive keys inside JavaScript's exact-integer range. Repaired works use
negative M keys; unsigned binary ID files use bit 63 as the M namespace tag.
"""
import re

IDENTITY_TAG = 1 << 63
MAX_IDENTITY_KEY = (1 << 53) - 1
_AUTHOR_ID = re.compile(r"^(?:https://openalex\.org/)?(A\d+)$|^(M\d+)$")


def bare_author_id(raw_id: str) -> str:
    match = _AUTHOR_ID.fullmatch(str(raw_id))
    if not match:
        raise ValueError(f"invalid author identity: {raw_id!r}")
    aid = match.group(1) or match.group(2)
    key = int(aid[1:])
    if aid[0] == "M" and not 0 < key <= MAX_IDENTITY_KEY:
        raise ValueError("Mapademic identity key must be positive and <= 2^53-1")
    if aid[0] == "A" and not 0 <= key < IDENTITY_TAG:
        raise ValueError("OpenAlex author key is outside the supported uint63 range")
    return aid[0] + str(key)


def author_key(raw_id: str) -> int:
    """Signed key in compact works extracts; namespace cannot collide."""
    aid = bare_author_id(raw_id)
    key = int(aid[1:])
    return -key if aid[0] == "M" else key


def encode_author_id(raw_id: str) -> int:
    """Unsigned binary record, compatible with all existing A-only files."""
    aid = bare_author_id(raw_id)
    return int(aid[1:]) | (IDENTITY_TAG if aid[0] == "M" else 0)


def decode_author_id(value: int) -> str:
    """Decode an unsigned binary record without losing its namespace."""
    value = int(value)
    if not 0 <= value < 1 << 64:
        raise ValueError("author identity record is outside uint64")
    aid = ("M" + str(value ^ IDENTITY_TAG)) if value & IDENTITY_TAG else "A" + str(value)
    return bare_author_id(aid)
