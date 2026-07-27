"""PolyNorm-style few-shot LLM text normalization for the Arabic TTS path.

Converts written Arabic (incl. mixed Arabic/English) into spoken form before
SILMA synthesis: digits -> words with agreement, "/" alternatives -> "أو",
ranges, percentages, entity expansion, and identifiers digit-by-digit.

Pipeline (`prepare()`), applied identically to both paths so the LLM and the
deterministic fallback agree on the hard cases:

    Arabic-Indic digits -> ASCII   (٢٢٠ -> 220)
    fixed entities                 (GDRFA, UAE, AED, ICP, VAT, Emirates ID)
    dates                          (15/03/2026 -> الخامس عشر من مارس ...)
    clock times                    (7:30 -> السابعة والنصف)
    identifiers                    (+971, A1234567, 784-1990-... -> digit-by-digit)

Everything the pre-pass resolves is already words by the time the LLM (or
num2words) sees it, so a date can never be read as a "/" alternatives pair and
a passport number can never be read as a quantity.

Safety net: a deterministic validator checks that every digit sequence in the
prepared source survives into the output (as digits, as quantity-words, or
digit-by-digit). Validation failure -> caller falls back to the num2words path,
so the LLM can never silently change a number.

Few-shot examples come from the validated gold set (docs/tts-norm-gold.jsonl,
ids 1,4,7,8,9,10,12,16); the remaining entries are held out for eval.
"""
import logging
import re
import time

from num2words import num2words

logger = logging.getLogger("uvicorn.error")

# ---- few-shot prompt ----
NORMALIZER_SYSTEM = (
    "أنت مُطبِّع نصوص لنظام تحويل النص إلى كلام. حوِّل النص المكتوب إلى صيغته "
    "المنطوقة بالعربية الفصحى دون تغيير المعنى أبدًا:\n"
    "- حوِّل كل الأرقام إلى كلمات عربية بالتشكيل النحوي الصحيح.\n"
    "- العملات: \"درهم\" حسب الإعراب، AED = درهم إماراتي.\n"
    "- الرمز \"/\" بين بديلين يُقرأ \"أو\".\n"
    "- النطاقات (8–96) تُقرأ \"من ... إلى ...\".\n"
    "- \"%\" تُقرأ \"في المئة\".\n"
    "- UAE = الإمارات، GDRFA = الإدارة العامة للإقامة وشؤون الأجانب.\n"
    "- أرقام الهويات والطلبات والملفات والجوازات والهواتف تُقرأ رقمًا رقمًا "
    "(2009607 = اثنان صفر صفر تسعة ستة صفر سبعة)، وليس ككمية. ميِّزها من "
    "السياق (رقم الطلب، رقم الملف، رقم الجواز...). الحروف داخلها تُقرأ بأسمائها.\n"
    "- الترتيب: بعد \"الطابق\" أو \"المبنى\" أو \"الدور\" يُقرأ الرقم ترتيبيًا "
    "(الطابق 3 = الطابق الثالث)، أما بعد \"رقم\" فيبقى عدديًا (بوابة رقم 4 = "
    "بوابة رقم أربعة).\n"
    "- التواريخ والأوقات وأرقام الهواتف تصلك محوَّلة إلى كلمات مسبقًا — اتركها "
    "كما هي ولا تُعِد صياغتها.\n"
    "- كل رقم في النص مرفق بتلميحين بين «»: صيغة الكمية وصيغة رقمًا رقمًا. "
    "انسخ الصيغة المناسبة للسياق وصحِّح إعرابها إن لزم — لا تحسب الأرقام "
    "بنفسك أبدًا، ولا تُظهر الأقواس «» أو محتواها غير المختار في الناتج.\n"
    "- قاعدة الاختيار: إذا تبع الرقمَ وحدةٌ (درهم، ساعة، يوم، سنة، شهر، %) "
    "فهو كمية دائمًا — استخدم صيغة الكمية. لا تستخدم صيغة رقمًا رقمًا إلا "
    "بعد سياق معرِّفات صريح (رقم الطلب، رقم الملف، رقم الجواز، رقم الهاتف، "
    "الرقم الموحد، الهوية).\n"
    "- اترك أي كلمات إنجليزية أخرى كما هي. لا تضف ولا تحذف معلومة.\n"
    "أعد النص المنطوق فقط دون أي شرح."
)

FEW_SHOT = [
    ("تبلغ تكلفة فيزا السياحة لمدة 30 يوم 220 درهمًا إماراتيًا خارج الدولة / 720 درهمًا داخل الدولة.",
     "تبلغ تكلفة فيزا السياحة لمدة ثلاثين يومًا مئتين وعشرين درهمًا إماراتيًا خارج الدولة، أو سبعمئة وعشرين درهمًا داخل الدولة."),
    ("الرسوم الإجمالية للتأشيرة السياحية متعددة الدخول لمدة 5 سنوات هي 713.5 درهمًا إماراتيًا (بإضافة 3000 درهم كوديعة قابلة للرد).",
     "الرسوم الإجمالية للتأشيرة السياحية متعددة الدخول لمدة خمس سنوات هي سبعمئة وثلاثة عشر درهمًا ونصف الدرهم، بإضافة ثلاثة آلاف درهم كوديعة قابلة للرد."),
    ("UAE تُعرف بكونها مركزاً اقتصادياً وسياحياً مهماً في منطقة الخليج.",
     "الإمارات تُعرف بكونها مركزًا اقتصاديًا وسياحيًا مهمًا في منطقة الخليج."),
    ("للحصول على الإقامة الذهبية Golden Visa يجب إثبات استثمار عقاري بقيمة 2,000,000 درهم.",
     "للحصول على الإقامة الذهبية Golden Visa يجب إثبات استثمار عقاري بقيمة مليوني درهم."),
    ("تضاف ضريبة القيمة المضافة بنسبة 5% إلى الرسوم.",
     "تضاف ضريبة القيمة المضافة بنسبة خمسة في المئة إلى الرسوم."),
    ("يجب أن تتراوح مدة التوقف بين 8 و96 ساعة في مطارات الدولة.",
     "يجب أن تتراوح مدة التوقف بين ثماني ساعات وست وتسعين ساعة في مطارات الدولة."),
    ("تستغرق المعالجة عادة 48 ساعة من تاريخ تقديم الطلب.",
     "تستغرق المعالجة عادة ثمانٍ وأربعين ساعة من تاريخ تقديم الطلب."),
    ("رقم طلبك هو 2009607.",
     "رقم طلبك هو اثنان صفر صفر تسعة ستة صفر سبعة."),
    ("رقم الملف الخاص بك هو 784023 ويجب تجديده خلال 30 يومًا.",
     "رقم الملف الخاص بك هو سبعة ثمانية أربعة صفر اثنان ثلاثة ويجب تجديده خلال ثلاثين يومًا."),
    ("للاستفسار يرجى الاتصال على الرقم 8005111.",
     "للاستفسار يرجى الاتصال على الرقم ثمانية صفر صفر خمسة واحد واحد واحد."),
    ("الملف المرجعي GD54321 قيد المراجعة حاليًا.",
     "الملف المرجعي جي دي خمسة أربعة ثلاثة اثنان واحد قيد المراجعة حاليًا."),
    ("يمكنك تقديم الطلب عبر موقع GDRFA الإلكتروني مقابل 100 درهم.",
     "يمكنك تقديم الطلب عبر موقع الإدارة العامة للإقامة وشؤون الأجانب الإلكتروني مقابل مئة درهم."),
    ("الطابق 3، بوابة رقم 4.",
     "الطابق الثالث، بوابة رقم أربعة."),
]


# ---------------------------------------------------------------- digit systems
# Arabic-Indic (٠-٩) and extended/Persian (۰-۹) numerals fold to ASCII up front.
# Everything downstream — _NUM_RE, _DIGIT_NAMES, num2words, the validator — is
# keyed on ASCII, so without this fold an Arabic-Indic identifier gets an empty
# digit-by-digit hint and can never validate.
_DIGIT_FOLD = str.maketrans("٠١٢٣٤٥٦٧٨٩" + "۰۱۲۳۴۵۶۷۸۹", "0123456789" * 2)


def fold_digits(text: str) -> str:
    return text.translate(_DIGIT_FOLD)


# ------------------------------------------------------------------- num2words
# num2words' Arabic output needs three fixes before it is speakable:
#   "مائة" -> "مئة" (matches the gold set's spelling)
#   " و " -> " و"   (it spaces the conjunction off its word)
#   nominative -> genitive for the duals that follow a preposition/عام
_NOMINATIVE_TO_GENITIVE = {
    "ألفان": "ألفين", "مئتان": "مئتين", "مليونان": "مليونين", "اثنان": "اثنين",
}


def _clean_ar(words: str) -> str:
    words = words.replace("مائة", "مئة")
    words = re.sub(r"\s+و\s+", " و", words)
    return re.sub(r"\s{2,}", " ", words).strip()


def _to_words(value: int) -> str:
    return _clean_ar(num2words(value, lang="ar"))


def _genitive(words: str) -> str:
    """Years follow a preposition ('عام ...'), so the whole compound inflects:
    'ألفان وستة وعشرون' -> 'ألفين وستة وعشرين'."""
    for nom, gen in _NOMINATIVE_TO_GENITIVE.items():
        words = re.sub(rf"\b{nom}\b", gen, words)
    return re.sub(r"(\w{2,})ون\b", r"\1ين", words)


def _decimal_words(int_part: str, frac_part: str) -> str:
    """num2words renders 1234.75 as 'ألف ... ,  خمس و سبعون' — a literal comma —
    and drops the integer part entirely for 0.5. Build decimals ourselves:
    integer as a quantity, fraction read digit-by-digit after 'فاصلة'."""
    whole = _to_words(int(int_part or 0))
    if not frac_part:
        return whole
    return f"{whole} فاصلة {_digit_names(frac_part)}"


# ------------------------------------------------------------------ dates/times
_AR_MONTHS = [
    "يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو",
    "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر",
]

_AR_ORDINALS = [
    "", "الأول", "الثاني", "الثالث", "الرابع", "الخامس", "السادس", "السابع",
    "الثامن", "التاسع", "العاشر", "الحادي عشر", "الثاني عشر", "الثالث عشر",
    "الرابع عشر", "الخامس عشر", "السادس عشر", "السابع عشر", "الثامن عشر",
    "التاسع عشر", "العشرون", "الحادي والعشرون", "الثاني والعشرون",
    "الثالث والعشرون", "الرابع والعشرون", "الخامس والعشرون", "السادس والعشرون",
    "السابع والعشرون", "الثامن والعشرون", "التاسع والعشرون", "الثلاثون",
    "الحادي والثلاثون",
]

# feminine ordinals — "الساعة" is feminine
_AR_HOURS = [
    "الثانية عشرة", "الواحدة", "الثانية", "الثالثة", "الرابعة", "الخامسة",
    "السادسة", "السابعة", "الثامنة", "التاسعة", "العاشرة", "الحادية عشرة",
    "الثانية عشرة",
]

# DD/MM/YYYY or DD-MM-YYYY (UAE convention), plus ISO YYYY-MM-DD
_DATE_DMY = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b")
_DATE_ISO = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_DATE_NAMED = re.compile(
    r"\b(\d{1,2})\s+(" + "|".join(_AR_MONTHS) + r")\b(?:\s+(\d{4}))?"
)
_TIME_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
_HOUR_PREFIX = re.compile(r"الساعة\s*$")
_DAYPART_NEARBY = re.compile(r"\s*(صباح|مساء|ظهر|عصر|ليل)")


def _spoken_date(day: int, month: int, year: int) -> str | None:
    if not (1 <= day <= 31 and 1 <= month <= 12):
        return None
    return (
        f"{_AR_ORDINALS[day]} من {_AR_MONTHS[month - 1]} "
        f"عام {_genitive(_to_words(year))}"
    )


def expand_dates(text: str) -> str:
    """Dates must resolve before anything else touches them: the '/' in
    15/03/2026 would otherwise hit the alternatives rule and be read as
    'fifteen or three or 2026'."""

    def dmy(m):
        return _spoken_date(int(m.group(1)), int(m.group(2)), int(m.group(3))) or m.group(0)

    def iso(m):
        return _spoken_date(int(m.group(3)), int(m.group(2)), int(m.group(1))) or m.group(0)

    def named(m):
        day, month = int(m.group(1)), _AR_MONTHS.index(m.group(2)) + 1
        if not 1 <= day <= 31:
            return m.group(0)
        out = f"{_AR_ORDINALS[day]} من {m.group(2)}"
        if m.group(3):
            out += f" عام {_genitive(_to_words(int(m.group(3))))}"
        return out

    text = _DATE_DMY.sub(dmy, _DATE_ISO.sub(iso, text))
    return _DATE_NAMED.sub(named, text)


def expand_times(text: str) -> str:
    """7:30 -> 'السابعة والنصف'. Without this the colon survives into speech
    and the minutes are read as a bare cardinal ('سبعة:ثلاثون')."""

    def repl(m):
        hour, minute = int(m.group(1)), int(m.group(2))
        named_hour = _AR_HOURS[hour % 12 if hour % 12 else 12]
        # "الساعة 18:00" already carries the word — don't say it twice
        spoken = named_hour if _HOUR_PREFIX.search(text[:m.start()]) else f"الساعة {named_hour}"
        if minute == 0:
            spoken += " تمامًا"
        elif minute == 15:
            spoken += " والربع"
        elif minute == 30:
            spoken += " والنصف"
        else:
            spoken += f" و{_to_words(minute)} دقيقة"
        # 24-hour clock: name the daypart, unless the text already does
        if hour >= 13 and not _DAYPART_NEARBY.match(text[m.end():]):
            spoken += " مساءً" if hour >= 18 else " بعد الظهر"
        return spoken

    return _TIME_RE.sub(repl, text)


# ----------------------------------------------------------------- identifiers
_LATIN_LETTER_NAMES = {
    "a": "إيه", "b": "بي", "c": "سي", "d": "دي", "e": "إي", "f": "إف",
    "g": "جي", "h": "إتش", "i": "آي", "j": "جي", "k": "كيه", "l": "إل",
    "m": "إم", "n": "إن", "o": "أو", "p": "بي", "q": "كيو", "r": "آر",
    "s": "إس", "t": "تي", "u": "يو", "v": "في", "w": "دبليو", "x": "إكس",
    "y": "واي", "z": "زد",
}

_PHONE_INTL = re.compile(r"\+\s?(\d[\d\s]*\d|\d)")
_LETTER_GLUED = re.compile(r"\b([A-Za-z]{1,4})(\d{4,})\b")
_HYPHEN_GROUPS = re.compile(r"\b\d+(?:-\d+)+\b")
# a bare long run is an identifier unless a unit follows it (2000000 درهم)
_LONG_RUN = re.compile(
    r"(?<![\d.,])\d{7,}(?!\d)(?!\.\d)(?!\s*(?:درهم|دينار|ريال|دولار|ألف|مليون))"
)


def _spell_latin(letters: str) -> str:
    return " ".join(_LATIN_LETTER_NAMES.get(c.lower(), c) for c in letters)


def _hyphen_groups(m):
    """784-1990-1234567-8 is a file number; 8-96 is a range. Digit count decides."""
    digits = m.group(0).replace("-", "")
    return _digit_names(digits) if len(digits) >= 7 else m.group(0)


def expand_identifiers(text: str) -> str:
    """Patterns that are structurally identifiers, regardless of context.

    The LLM path infers identifiers from surrounding words ('رقم الطلب'), but
    the deterministic fallback has no such notion — untouched, it reads +971 as
    'nine hundred seventy-one' and A1234567 as one-point-two million.
    Comma-grouped numbers (2,000,000) and decimals are excluded: those are
    quantities by construction.
    """
    text = _PHONE_INTL.sub(lambda m: _digit_names(re.sub(r"\s", "", m.group(1))), text)
    text = _LETTER_GLUED.sub(
        lambda m: f"{_spell_latin(m.group(1))} {_digit_names(m.group(2))}", text
    )
    text = _HYPHEN_GROUPS.sub(_hyphen_groups, text)
    return _LONG_RUN.sub(lambda m: _digit_names(m.group(0)), text)


# Ordinals are context-bound: "الطابق 3" is الثالث, but "بوابة رقم 4" stays
# أربعة. Only an explicit trigger word converts — anything else is left to the
# LLM's ordinal rule rather than guessed at here.
_ORDINAL_TRIGGER = re.compile(r"\b(الطابق|الدور|المبنى)\s+(\d{1,2})\b")


def expand_ordinals(text: str) -> str:
    def repl(m):
        n = int(m.group(2))
        return f"{m.group(1)} {_AR_ORDINALS[n]}" if 1 <= n <= 31 else m.group(0)

    return _ORDINAL_TRIGGER.sub(repl, text)


# ------------------------------------------------------------- fixed entities
# Expansions are a dictionary lookup, not a judgment call — apply them
# deterministically before any model sees the text (the 8B ignores the
# expansion instruction ~half the time). Longest first so "Emirates ID" wins
# over a bare "ID".
_ENTITY_MAP = [
    (re.compile(r"\bGDRFA\b"), "الإدارة العامة للإقامة وشؤون الأجانب"),
    (re.compile(r"\bEmirates\s+ID\b", re.I), "الهوية الإماراتية"),
    (re.compile(r"\bICP\b"), "الهيئة الاتحادية للهوية والجنسية"),
    (re.compile(r"\bVAT\b"), "ضريبة القيمة المضافة"),
    (re.compile(r"\bUAE\b"), "الإمارات"),
    (re.compile(r"\bAED\b"), "درهم إماراتي"),
]


def expand_entities(text: str) -> str:
    for pat, repl in _ENTITY_MAP:
        text = pat.sub(repl, text)
    return text


def prepare(text: str) -> str:
    """Deterministic pre-pass shared by both paths. Order matters: dates and
    times claim their digits before the identifier rules can, and both run
    before any generic number handling."""
    text = fold_digits(text)
    text = expand_entities(text)
    text = expand_dates(text)
    text = expand_times(text)
    text = expand_ordinals(text)
    text = expand_identifiers(text)
    return text.replace("%", " في المئة").replace("٪", " في المئة")


# ---- PDS-lite: annotate digit sequences with both candidate renderings so
# the LLM chooses and inflects instead of computing (small models hallucinate
# arithmetic — observed: 96 -> "تسع وتسعين", Hebrew script leakage on 1,140).
def _digit_names(digits: str) -> str:
    return " ".join(_DIGIT_NAMES[d][0] for d in digits if d in _DIGIT_NAMES)


def annotate(text: str) -> str:
    def repl(m):
        raw = m.group(0)
        plain = raw.replace(",", "")
        int_part, _, frac_part = plain.partition(".")
        try:
            qty = _decimal_words(int_part, frac_part)
        except Exception:
            return raw
        return f"{raw}«ككمية: {qty}؛ رقمًا رقمًا: {_digit_names(int_part)}»"

    return _NUM_RE.sub(repl, text)


def _build_messages(text: str):
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

    msgs = [SystemMessage(content=NORMALIZER_SYSTEM)]
    for src, spoken in FEW_SHOT:
        msgs.append(HumanMessage(content=annotate(prepare(src))))
        msgs.append(AIMessage(content=spoken))
    msgs.append(HumanMessage(content=annotate(text)))
    return msgs


# ---- deterministic digit-preservation validator ----
_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")

_DIGIT_NAMES = {
    "0": ["صفر"], "1": ["واحد", "واحدة"], "2": ["اثنان", "اثنين", "إثنان"],
    "3": ["ثلاثة", "ثلاث"], "4": ["أربعة", "أربع"], "5": ["خمسة", "خمس"],
    "6": ["ستة", "ست"], "7": ["سبعة", "سبع"], "8": ["ثمانية", "ثمان", "ثماني"],
    "9": ["تسعة", "تسع"],
}


def _norm_ar(s: str) -> str:
    """Orthographic normalization so case/spelling variants compare equal:
    strip diacritics, unify مائة/مئة, unify ـون/ـان/ـين endings on number words."""
    s = re.sub(r"[ً-ْٰ]", "", s)  # diacritics
    s = s.replace("مائة", "مئة").replace("مئتان", "مئتين").replace("ألفان", "ألفين")
    s = s.replace("مليونان", "مليونين").replace("مليوني", "مليونين")
    s = re.sub(r"\bمليونا\b", "مليونين", s)
    s = re.sub(r"\bألفا\b", "ألفين", s).replace("مئتا ", "مئتين ")
    s = re.sub(r"(\w{2,})ون\b", r"\1ين", s)  # عشرون -> عشرين etc.
    s = re.sub(r"\s+و\s+", " و", s)  # num2words puts spaces around و
    s = s.replace("إ", "ا").replace("أ", "ا").replace("آ", "ا")
    s = re.sub(r"\bثماني?ة?\b", "ثمان", s)  # ثمانية/ثماني/ثمان -> one form
    return s


def _quantity_words(value_str: str) -> str | None:
    try:
        value = int(value_str)
    except ValueError:
        return None
    try:
        return _norm_ar(_to_words(value))
    except Exception:
        return None


def _digit_by_digit_present(digits: str, out_norm: str) -> bool:
    """Do the digit names appear in order (possibly with connectors between)?"""
    pos = 0
    for d in digits:
        candidates = [_norm_ar(n) for n in _DIGIT_NAMES.get(d, [])]
        hits = [out_norm.find(c, pos) for c in candidates]
        hits = [h for h in hits if h >= 0]
        if not hits:
            return False
        pos = min(hits) + 1
    return True


def validate_numbers(source: str, output: str) -> bool:
    """Every digit sequence in source must survive in output: as digits, as
    quantity-words (integer part for decimals), or digit-by-digit."""
    out_norm = _norm_ar(output)
    for m in _NUM_RE.finditer(source):
        raw = m.group(0)
        plain = raw.replace(",", "")
        if plain in output.replace(",", ""):
            continue  # digits kept verbatim
        int_part = plain.split(".")[0]
        words = _quantity_words(int_part)
        if words is not None:
            # accept if most significant tokens of the words rendering appear;
            # tolerate gender/agreement variants (ستة/ست، ثمانية/ثمان)
            tokens = [t for t in words.split() if len(t) > 2 and t != "درهم"]
            if tokens and all(
                t in out_norm or (t.endswith("ة") and t[:-1] in out_norm)
                for t in tokens
            ):
                continue
        if _digit_by_digit_present(int_part, out_norm):
            continue
        logger.info(f"[tts-norm] validator MISS for {raw!r}")
        return False
    return True


# ---- fallback: the deterministic path ----
def num2words_normalize(text: str) -> str:
    text = prepare(text)

    def repl(m):
        raw = m.group(0).replace(",", "")
        int_part, _, frac_part = raw.partition(".")
        try:
            return _decimal_words(int_part, frac_part)
        except (ValueError, TypeError):
            return m.group(0)

    return _NUM_RE.sub(repl, text)


def llm_normalize(text: str) -> tuple[str, str]:
    """Returns (normalized_text, route). Falls back to num2words when the LLM
    output fails digit-preservation or the call errors."""
    from .backend_setup import clarifier_llm
    from .system_prompt_setup import NO_THINK

    prepared = prepare(text)
    t0 = time.perf_counter()
    try:
        msgs = _build_messages(prepared)
        msgs[0].content += NO_THINK
        result = str(clarifier_llm.invoke(msgs).content).strip()
        if result and validate_numbers(prepared, result):
            logger.info(f"[timing] tts-norm llm: {time.perf_counter() - t0:.1f}s | ok")
            return result, "llm"
        logger.info(
            f"[timing] tts-norm llm: {time.perf_counter() - t0:.1f}s | "
            "validation failed -> num2words fallback"
        )
    except Exception as e:
        logger.error(f"[tts-norm] llm error -> num2words fallback: {e}")
    return num2words_normalize(text), "num2words-fallback"
