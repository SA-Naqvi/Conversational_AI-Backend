"""
Token counting and budget management for LLM prompts.
Uses a simple word-based approximation (≈1.3 tokens per word for English).
"""


# Approximate tokens per word for English text with Qwen tokenizer
TOKENS_PER_WORD = 1.3


def count_tokens(text: str) -> int:
    """
    Approximate token count for a text string.
    Uses word count * 1.3 as a rough approximation.
    """
    if not text:
        return 0
    words = text.split()
    return int(len(words) * TOKENS_PER_WORD)


def trim_history(history: list, max_tokens: int) -> list:
    """
    Trim conversation history to fit within a token budget.
    Keeps the most recent messages, removing oldest first.

    Args:
        history: list of {"role": ..., "content": ...} dicts
        max_tokens: maximum total tokens for history

    Returns:
        Trimmed history list (most recent entries preserved)
    """
    if not history:
        return []

    total = 0
    trimmed = []

    # Iterate from most recent to oldest
    for msg in reversed(history):
        content = msg.get("content", "")
        msg_tokens = count_tokens(content)
        if total + msg_tokens > max_tokens:
            break
        total += msg_tokens
        trimmed.append(msg)

    # Reverse back to chronological order
    return list(reversed(trimmed))


def check_budget(
    system_tokens: int,
    state_tokens: int,
    history_tokens: int,
    user_tokens: int,
    max_context: int = 4096,
    response_reserve: int = 1596,
) -> dict:
    """
    Check if the current prompt fits within the token budget.

    Returns:
        {
            "fits": bool,
            "total_used": int,
            "remaining_for_response": int,
            "over_budget_by": int  (0 if fits)
        }
    """
    total_used = system_tokens + state_tokens + history_tokens + user_tokens
    available = max_context - total_used

    return {
        "fits": available >= response_reserve,
        "total_used": total_used,
        "remaining_for_response": max(available, 0),
        "over_budget_by": max(response_reserve - available, 0),
    }
