"""Sentence chunking for streamed synthesis.

SILMA (and the Kokoro path) synthesize a whole utterance before returning
anything, so a 20-second answer costs a 20-second-ish wait before the first
sound. Splitting the answer into sentences and synthesizing them in sequence
lets playback start after the first clip while the rest is still being made.

The trade-off is prosody: too-short clips sound clipped and lose sentence
intonation, and each clip pays the model's fixed per-call cost. So fragments
are merged up to MIN_CHARS, and only genuinely long sentences are split at a
comma to stay under MAX_CHARS.
"""
import re

MIN_CHARS = 60        # merge shorter fragments into the next one
FIRST_MIN_CHARS = 24  # the opening clip gates time-to-first-audio: keep it small
MAX_CHARS = 240       # above this, split at a clause boundary
FIRST_MAX_CHARS = 120 # tighter ceiling on the opening clip

# Arabic full stop/question/semicolon alongside their Latin counterparts.
# Note this runs AFTER tts_normalizer.prepare(), so decimal points, dates and
# identifiers are already words — a bare "." here is genuinely a sentence end.
_SENTENCE_END = re.compile(r"(?<=[.!?؟۔…])\s+|\n+")
_CLAUSE_END = re.compile(r"(?<=[,،;؛:])\s+")


def _split_long(chunk: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Break an over-long sentence at clause boundaries, keeping each piece
    under max_chars where the punctuation allows it."""
    if len(chunk) <= max_chars:
        return [chunk]
    out, current = [], ""
    for piece in _CLAUSE_END.split(chunk):
        if current and len(current) + len(piece) + 1 > max_chars:
            out.append(current.strip())
            current = piece
        else:
            current = f"{current} {piece}".strip() if current else piece
    if current.strip():
        out.append(current.strip())
    return out


def split_for_streaming(
    text: str,
    min_chars: int = MIN_CHARS,
    max_chars: int = MAX_CHARS,
    first_min_chars: int = FIRST_MIN_CHARS,
    first_max_chars: int = FIRST_MAX_CHARS,
) -> list[str]:
    """Ordered clips to synthesize. Concatenating them reproduces `text`
    (whitespace aside) — no content is added or dropped."""
    text = (text or "").strip()
    if not text:
        return []

    sentences = [s.strip() for s in _SENTENCE_END.split(text) if s and s.strip()]

    # Merge forward so no clip is a stub, then split anything still too long.
    # The opening clip uses a lower bar: nothing plays until it is synthesized,
    # so a short first sentence is worth more than its slightly flatter prosody.
    merged: list[str] = []
    for sentence in sentences:
        threshold = first_min_chars if len(merged) == 1 else min_chars
        if merged and len(merged[-1]) < threshold:
            merged[-1] = f"{merged[-1]} {sentence}"
        else:
            merged.append(sentence)

    # a trailing stub has nothing to merge into — fold it backwards instead
    # (pop first: `merged[-2] = ... merged.pop()` resolves the index after the
    # pop has already shortened the list)
    if len(merged) > 1 and len(merged[-1]) < min_chars:
        tail = merged.pop()
        merged[-1] = f"{merged[-1]} {tail}"

    # A single long opening sentence would stall playback just as badly as no
    # chunking at all, so the first clip gets a tighter ceiling than the rest.
    out: list[str] = []
    for index, chunk in enumerate(merged):
        ceiling = first_max_chars if index == 0 else max_chars
        out.extend(_split_long(chunk, ceiling))
    return out
