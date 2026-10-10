"""[4a] Safety input gate (ARCHITECTURE.md §6): mask PII, then detect threats on the masked text. No LLM."""

import re

from copilot.schemas import ResponseLLMOutput, SafetyCheck

# --- Masking patterns: 05_responsible_ai_policy.md#pii-and-payment-data-masking ---

# 13-19 digits, spaces/hyphens allowed between digits. Redact even if Luhn fails (safer).
CARD_PATTERN = re.compile(r"(?<!\d)\d(?:[ -]?\d){12,18}(?!\d)")
# Only mask when digits follow: "PIN retry failed" is a POS issue, "PIN is 4321" is a secret.
NUMERIC_SECRET_PATTERN = re.compile(r"\b(cvv|pin|otp)(\s*(?:is|:|=|-)?\s*)\d{3,8}\b", re.IGNORECASE)
# Needs "is", ":" or "=" so "password reset failed" is not masked; the value can be any non-space text.
PASSWORD_PATTERN = re.compile(r"\b(password|passcode)(\s*(?:is|:|=)\s*)\S+", re.IGNORECASE)
# Optional +91, 10 digits starting 6-9, optionally split 5+5 ("98765 43210").
PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+91[\s-]?)?([6-9]\d{4})[\s-]?(\d{5})(?!\d)")
AADHAAR_PATTERN = re.compile(r"(?<!\d)\d{4}\s?\d{4}\s?\d{4}(?!\d)")
# Case-sensitive on purpose: PAN is printed in capitals; store ids and SKUs never have this shape.
PAN_PATTERN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
EMAIL_PATTERN = re.compile(r"\b([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")

# --- Threat patterns: 05_responsible_ai_policy.md#prompt-injection ---

INJECTION_PATTERNS = [
    r"\bignore\s+(?:\w+\s+){0,2}(?:instructions|policies|rules|guidelines)\b",
    # Deliberate change from spec: bare "disregard" blocked "customer said disregard the earlier order".
    # Block only when it targets instructions or controls.
    r"\bdisregard\s+(?:\w+\s+){0,3}(?:instructions|previous|prior|above|rules|polic(?:y|ies)|safety|system)\b",
    # Deliberate change from spec: bare "bypass" blocked "bypass the card reader and take cash",
    # a legitimate POS workaround. Block only when the target is a policy or safety control.
    r"\bbypass\w*\s+(?:\w+\s+){0,3}(?:polic(?:y|ies)|rules|safety|security|guardrails?|filters?)\b",
    r"\bjailbreak",
    r"\byou are now\b",
    r"\bdeveloper mode\b",
    r"\breveal\s+(?:the\s+|your\s+)?(?:hidden|system)\s+prompt\b",
    r"\bhidden\s+(?:judge\s+)?tests?\b",
    r"\bdo not cite\b",
    r"\binvent\s+a\s+policy\b",
]

# Request verb followed within ~8 tokens by the target (keyword window).
WINDOW = r"\b(?:\s+\S+){0,8}?\s+"
RESTRICTED_VERBS = r"\b(?:show|give|reveal|print|list|export|send|display|share|dump|tell)"
RESTRICTED_TARGETS = (
    r"(?:supplier\s+(?:contract\s+)?prices?|contract\s+prices?|full\s+customer\s+database"
    r"|customer\s+(?:phone\s+numbers|records|data)|employee\s+records|admin\s+password|credentials)\b"
)

# Deliberate change from spec: "tell" is excluded so "tell the customer to re-enter their PIN" is not blocked.
PAYMENT_VERBS = r"\b(?:show|give|reveal|print|list|export|send|display|share|dump)"
# "type/enter/write/paste" only count when aimed at the chat, so "enter PIN on the terminal" stays legit.
TYPE_VERBS = r"\b(?:type|enter|write|paste)"
INTO_CHAT = r"(?:\s+\S+){0,4}?\s+(?:in|into)\s+(?:the\s+)?(?:chat|here)\b"
# "PIN pad", "PIN retry" etc. name a device or process, not the secret itself.
PAYMENT_TARGETS = (
    r"(?:card\s+numbers?|cvv|otp|payment\s+tokens?|banking\s+details"
    r"|pin(?!\s*-?\s*(?:pad|entry|retry|retries|attempts?|prompt|screen|reset|change)))\b"
)


# Phone match -> "******" + last 4 digits.
def _mask_phone(match: re.Match) -> str:
    digits = match.group(1) + match.group(2)
    return "******" + digits[-4:]


# Email match -> local part as first char + stars + last char, domain kept.
def _mask_email(match: re.Match) -> str:
    local, domain = match.group(1), match.group(2)
    if len(local) < 3:
        return local[0] + "*@" + domain
    return local[0] + "*" * (len(local) - 2) + local[-1] + "@" + domain


# Applies one pattern; records `label` in pii_found when it matched. Returns the new text.
def _apply_mask(text: str, pattern: re.Pattern, replacement, label: str, pii_found: list[str]) -> str:
    masked, count = pattern.subn(replacement, text)
    if count and label not in pii_found:
        pii_found.append(label)
    return masked


# Raw user text -> (masked text, list of PII types found). Order matters: card first so its
# digits are gone before the shorter phone/Aadhaar patterns run.
def mask_pii(text: str) -> tuple[str, list[str]]:
    pii_found: list[str] = []
    text = _apply_mask(text, CARD_PATTERN, "[REDACTED_CARD]", "card", pii_found)
    text = _apply_mask(text, NUMERIC_SECRET_PATTERN, r"\1\2[REDACTED]", "secret", pii_found)
    text = _apply_mask(text, PASSWORD_PATTERN, r"\1\2[REDACTED]", "password", pii_found)
    text = _apply_mask(text, PHONE_PATTERN, _mask_phone, "phone", pii_found)
    text = _apply_mask(text, AADHAAR_PATTERN, "[REDACTED]", "aadhaar", pii_found)
    text = _apply_mask(text, PAN_PATTERN, "[REDACTED]", "pan", pii_found)
    text = _apply_mask(text, EMAIL_PATTERN, _mask_email, "email", pii_found)
    return text, pii_found


# Masked text -> threat type ("none" if clean). Injection is checked first because it is the
# most severe and decides the Security routing (ARCHITECTURE.md §6 blocked outcome).
def detect_threat(masked_text: str) -> str:
    lowered = masked_text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            return "prompt_injection"
    if re.search(RESTRICTED_VERBS + WINDOW + RESTRICTED_TARGETS, lowered):
        return "restricted_data_request"
    if re.search(PAYMENT_VERBS + WINDOW + PAYMENT_TARGETS, lowered):
        return "payment_data_request"
    if re.search(TYPE_VERBS + WINDOW + PAYMENT_TARGETS + INTO_CHAT, lowered):
        return "payment_data_request"
    return "none"


BLOCK_REASONS = {
    "prompt_injection": "Request tries to override the copilot's instructions or policies.",
    "restricted_data_request": "Request asks for restricted business or personal data.",
    "payment_data_request": "Request asks for payment card or credential data.",
}


# Raw user text -> SafetyCheck. Must run before any LLM call, cache write or log write (AGENTS.md rule 2).
def check_input(text: str) -> SafetyCheck:
    masked_text, pii_found = mask_pii(text)
    threat_type = detect_threat(masked_text)
    return SafetyCheck(
        masked_text=masked_text,
        pii_found=pii_found,
        blocked=threat_type != "none",
        threat_type=threat_type,
        block_reason=BLOCK_REASONS.get(threat_type),
    )


# --- Output gate: ARCHITECTURE.md §6 [5], 05_responsible_ai_policy.md#human-in-the-loop ---

# Claims that an approval or action already happened. Deliberate narrowing of the spec regex: bare "approved"
# also blocked safe wording like "manager-approved transfer review".
COMPLETED_ACTION_PATTERN = re.compile(
    r"\b(i approve|i have approved|(?:refund|return|claim|transfer|exchange) (?:is|has been) approved"
    r"|has been refunded|refund (?:is |has been )?processed|claim (?:has been )?submitted"
    r"|is safe to use|guaranteed)\b", re.IGNORECASE)
MAX_ANSWER_ACTIONS = 5


# Draft citations + allowed set -> only allowed ids, in draft order, deduplicated.
def allowed_only(citations: list[str], allowed_citations: list[str]) -> list[str]:
    kept = []
    for citation in citations:
        if citation in allowed_citations and citation not in kept:
            kept.append(citation)
    return kept


# LLM draft + allowed citations -> cleaned draft, or None when the caller must use the template fallback
# (empty answer or a claim that an approval/action already happened).
def check_output(draft: ResponseLLMOutput, allowed_citations: list[str]) -> ResponseLLMOutput | None:
    answer = mask_pii(draft.answer.strip())[0]
    if not answer or COMPLETED_ACTION_PATTERN.search(answer):
        return None
    next_actions = []
    for action in draft.next_actions[:MAX_ANSWER_ACTIONS]:
        if action.strip() and not COMPLETED_ACTION_PATTERN.search(action):
            next_actions.append(mask_pii(action.strip())[0])
    follow_up = mask_pii(draft.follow_up_question)[0] if draft.follow_up_question else None
    return ResponseLLMOutput(answer=answer, next_actions=next_actions, follow_up_question=follow_up,
                             citations=allowed_only(draft.citations, allowed_citations))
