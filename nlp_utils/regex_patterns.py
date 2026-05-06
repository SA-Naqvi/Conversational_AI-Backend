"""
Compiled regex patterns for extracting medical data from patient messages.
"""
import re

# ============ Pain Level Patterns ============
# Match direct numeric pain levels: "my pain is 7", "pain level 8", "about a 6"
PAIN_DIRECT = re.compile(
    r'(?:pain\s*(?:is|level|score|at|of|:)?\s*(?:about|around|roughly)?\s*)'
    r'(\d{1,2})(?:\s*(?:/\s*10|out\s*of\s*10))?',
    re.IGNORECASE
)

# Match standalone numbers in pain context: "it's a 7", "about a 6"
PAIN_STANDALONE = re.compile(
    r'(?:(?:it\'?s|about|around|roughly|maybe)\s+(?:a\s+)?)?'
    r'(\d{1,2})\s*(?:/\s*10|out\s*of\s*10)',
    re.IGNORECASE
)

# Match word-based pain descriptors
PAIN_DESCRIPTORS = {
    "no pain": 0,
    "pain free": 0,
    "painless": 0,
    "minimal": 1,
    "very mild": 2,
    "mild": 3,
    "slight": 3,
    "moderate": 5,
    "significant": 6,
    "severe": 8,
    "very severe": 9,
    "excruciating": 10,
    "unbearable": 10,
    "worst pain": 10,
    "worst possible": 10,
}

# Relative pain change indicators
PAIN_WORSENED = re.compile(
    r'\b(?:worse|worsened|worsening|increased|increasing|higher|more\s+pain'
    r'|gone\s+up|getting\s+worse|intensified|aggravated)\b',
    re.IGNORECASE
)

PAIN_IMPROVED = re.compile(
    r'\b(?:better|improved|improving|decreased|decreasing|lower|less\s+pain'
    r'|gone\s+down|getting\s+better|relieved|subsided|eased)\b',
    re.IGNORECASE
)

PAIN_SAME = re.compile(
    r'\b(?:same|unchanged|no\s+change|stable|constant|hasn\'?t\s+changed)\b',
    re.IGNORECASE
)

# ============ Temperature Patterns ============
# Match Fahrenheit: "101.5F", "101.5°F", "101.5 degrees", "fever of 101.5"
TEMP_FAHRENHEIT = re.compile(
    r'(\d{2,3}(?:\.\d{1,2})?)\s*°?\s*(?:[fF](?:ahrenheit)?|degrees?\s*[fF]?)',
    re.IGNORECASE
)

# Match Celsius: "38.5C", "38.5°C"
TEMP_CELSIUS = re.compile(
    r'(\d{2}(?:\.\d{1,2})?)\s*°?\s*[cC](?:elsius)?',
    re.IGNORECASE
)

# Match standalone temperature-like numbers (contextual): "temperature is 101.5"
TEMP_CONTEXTUAL = re.compile(
    r'(?:temp(?:erature)?\s*(?:is|of|at|:)?\s*)'
    r'(\d{2,3}(?:\.\d{1,2})?)',
    re.IGNORECASE
)

# Match fever mentions
TEMP_FEVER_WORDS = re.compile(
    r'\b(?:fever|feverish|febrile|burning\s+up|feel(?:ing)?\s+hot)\b',
    re.IGNORECASE
)

# ============ Date Patterns ============
# Match full dates: "March 5, 2026", "3/5/2026", "2026-03-05"
DATE_FULL = re.compile(
    r'\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\b'
    r'|'
    r'\b(\d{4}[/\-]\d{1,2}[/\-]\d{1,2})\b'
    r'|'
    r'\b((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?'
    r'|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?'
    r'|Dec(?:ember)?)\s+\d{1,2}(?:\s*,?\s*\d{4})?)\b',
    re.IGNORECASE
)

# Match relative dates: "2 weeks ago", "3 days ago", "last Monday"
DATE_RELATIVE_AGO = re.compile(
    r'(\d+)\s*(day|week|month)s?\s*ago',
    re.IGNORECASE
)

DATE_RELATIVE_LAST = re.compile(
    r'last\s*(monday|tuesday|wednesday|thursday|friday|saturday|sunday|week|month)',
    re.IGNORECASE
)

DATE_YESTERDAY = re.compile(r'\byesterday\b', re.IGNORECASE)
DATE_TODAY = re.compile(r'\btoday\b', re.IGNORECASE)

# ============ Surgery Type Patterns ============
KNOWN_SURGERIES = [
    "knee replacement", "hip replacement", "shoulder replacement",
    "knee surgery", "hip surgery", "shoulder surgery",
    "appendectomy", "appendix removal",
    "hernia repair", "hernia surgery",
    "gallbladder removal", "cholecystectomy",
    "c-section", "cesarean", "caesarean",
    "heart bypass", "bypass surgery", "cabg",
    "acl repair", "acl reconstruction", "acl surgery",
    "rotator cuff", "rotator cuff repair",
    "back surgery", "spinal fusion", "spine surgery",
    "tonsillectomy", "tonsil removal",
    "cataract surgery", "cataract removal",
    "wisdom teeth", "tooth extraction", "dental surgery",
    "arthroscopy", "arthroscopic surgery",
    "laparoscopy", "laparoscopic surgery",
    "mastectomy", "lumpectomy",
    "hysterectomy",
    "prostatectomy",
    "colectomy",
    "thyroidectomy",
    "carpal tunnel",
    "gastric bypass", "bariatric surgery",
]

# Build a combined regex from known surgeries
SURGERY_TYPE_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(s) for s in KNOWN_SURGERIES) + r')\b',
    re.IGNORECASE
)

# ============ Name Patterns ============
# Simple name extraction: "my name is ...", "I'm ...", "call me ..."
NAME_PATTERN = re.compile(
    r'(?:(?:my\s+)?name\s+is|i\'?m|call\s+me|this\s+is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
    re.IGNORECASE
)
