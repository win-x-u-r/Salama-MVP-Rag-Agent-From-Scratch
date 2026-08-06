"""Sentence chunking for streamed synthesis.

SILMA (and the Kokoro path) synthesize a whole utterance before returning
anything, so a 20-second answer costs a 20-second-ish wait before the first
sound. Splitting the answer into clips and synthesizing them in sequence lets
playback start after the first clip while the rest is still being made.

Two stages, deliberately separate:

  1. SEGMENT into sentences — either the regex splitter or spaCy's multilingual
     `xx_sent_ud_sm` senter, selected by TTS_SEGMENTER. The spaCy path is
     best-effort: any import/load failure falls back to regex, so the app never
     hard-depends on the model being installed.
  2. SIZE the sentences into clips — merge stubs, then enforce a ceiling that
     no clip may exceed. This runs identically whichever segmenter produced the
     sentences, so the max-length guarantee holds on both paths.

The trade-off in stage 2 is prosody: too-short clips sound clipped and lose
sentence intonation, and each clip pays the model's fixed per-call cost.
"""
import logging
import os
import re

logger = logging.getLogger("uvicorn.error")

MIN_CHARS = 60        # merge shorter fragments into the next one
FIRST_MIN_CHARS = 24  # the opening clip gates time-to-first-audio: keep it small
MAX_CHARS = 240       # above this, split at a clause boundary
FIRST_MAX_CHARS = 120 # tighter ceiling on the opening clip

SPACY_MODEL = os.getenv("TTS_SPACY_MODEL", "xx_sent_ud_sm")

# Arabic full stop/question/semicolon alongside their Latin counterparts.
# Note this runs AFTER tts_normalizer.prepare(), so decimal points, dates and
# identifiers are already words — a bare "." here is genuinely a sentence end.
_SENTENCE_END = re.compile(r"(?<=[.!?؟۔…])\s+|\n+")
_CLAUSE_END = re.compile(r"(?<=[,،;؛:])\s+")

# Clause openers, used only when a chunk is over the ceiling and has no clause
# punctuation left to cut at. The `و\w{3,}` arm catches the prefixed waw, which
# is how Arabic actually starts most coordinate clauses; it can misfire on a
# word whose root simply begins with waw (وقت، وزارة), but the alternative at
# this point is a blind cut mid-phrase, so a near-miss boundary still wins.
_COORD_WORDS = ("ثم", "أو", "لكن", "بينما", "كما", "حيث", "أما",
                "and", "or", "but", "while", "then")
_COORDINATOR = re.compile(
    r"\s+(?=(?:" + "|".join(_COORD_WORDS) + r")\s)|\s+(?=و\w{3,})"
)


# ------------------------------------------------------------------ segmenters
_nlp = None
_spacy_failed = False


def _load_spacy():
    """Lazily load the senter. Returns None (once, loudly) if unavailable —
    callers fall back to the regex splitter."""
    global _nlp, _spacy_failed
    if _nlp is not None or _spacy_failed:
        return _nlp
    try:
        import time
        import spacy

        t0 = time.perf_counter()
        _nlp = spacy.load(SPACY_MODEL)
        logger.info(
            f"[tts-chunker] loaded {SPACY_MODEL} in {time.perf_counter() - t0:.1f}s "
            f"| pipeline={_nlp.pipe_names}"
        )
    except Exception as e:
        _spacy_failed = True
        logger.warning(
            f"[tts-chunker] spaCy segmenter unavailable ({e}) -> using regex"
        )
    return _nlp


def warmup() -> None:
    """Pay the model load at startup instead of on the first spoken answer.
    Safe to call regardless of which segmenter is configured."""
    if os.getenv("TTS_SEGMENTER", "regex").lower() == "spacy":
        _load_spacy()


def _sentences_regex(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s and s.strip()]


def _sentences_spacy(text: str) -> list[str] | None:
    nlp = _load_spacy()
    if nlp is None:
        return None
    try:
        return [s.text.strip() for s in nlp(text).sents if s.text.strip()]
    except Exception as e:
        logger.warning(f"[tts-chunker] spaCy segmentation failed ({e}) -> regex")
        return None


def _segment(text: str) -> tuple[list[str], str]:
    """(sentences, name of the segmenter that actually produced them)."""
    if os.getenv("TTS_SEGMENTER", "regex").lower() == "spacy":
        sentences = _sentences_spacy(text)
        if sentences:
            return sentences, "spacy"
    return _sentences_regex(text), "regex"


# ----------------------------------------------------------------- clip sizing
def _accumulate(pieces: list[str], max_chars: int) -> list[str]:
    """Glue pieces back together, starting a new clip whenever the next piece
    would push past the ceiling."""
    out: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) + 1 > max_chars:
            out.append(current.strip())
            current = piece
        else:
            current = f"{current} {piece}".strip() if current else piece
    if current.strip():
        out.append(current.strip())
    return out


def _hard_wrap(chunk: str, max_chars: int) -> list[str]:
    """Last resort: break on whitespace. Ugly prosody, but it is what makes the
    ceiling a guarantee rather than a hope."""
    return _accumulate(chunk.split(), max_chars) or [chunk]


def _split_long(chunk: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Bring a chunk under max_chars, trying the least damaging cut first:
    clause punctuation, then clause openers, then raw whitespace."""
    if len(chunk) <= max_chars:
        return [chunk]

    out: list[str] = []
    for piece in _accumulate(_CLAUSE_END.split(chunk), max_chars):
        if len(piece) <= max_chars:
            out.append(piece)
            continue
        for sub in _accumulate(_COORDINATOR.split(piece), max_chars):
            out.extend([sub] if len(sub) <= max_chars else _hard_wrap(sub, max_chars))
    return out


def chunk(
    text: str,
    min_chars: int = MIN_CHARS,
    max_chars: int = MAX_CHARS,
    first_min_chars: int = FIRST_MIN_CHARS,
    first_max_chars: int = FIRST_MAX_CHARS,
) -> tuple[list[str], str]:
    """(clips to synthesize in order, segmenter used).

    Concatenating the clips reproduces `text` whitespace-aside — no content is
    added or dropped — and no clip exceeds its ceiling.
    """
    text = (text or "").strip()
    if not text:
        return [], "none"

    sentences, segmenter = _segment(text)

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
    for index, piece in enumerate(merged):
        out.extend(_split_long(piece, first_max_chars if index == 0 else max_chars))
    return out, segmenter


def split_for_streaming(text: str, **kwargs) -> list[str]:
    """Clips only — kept so callers that don't care which segmenter ran can
    ignore the metadata."""
    return chunk(text, **kwargs)[0]
