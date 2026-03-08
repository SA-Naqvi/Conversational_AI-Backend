"""
Prompt caching layer.
Caches the rendered system prompt content (persona + patient state)
to avoid re-building identical prompts on every turn.
"""
import hashlib
import time
from typing import Optional

from config import PROMPT_CACHE_TTL


class PromptCache:
    """Simple in-memory prompt cache with TTL expiry."""

    def __init__(self, ttl: int = None):
        self.ttl = ttl or PROMPT_CACHE_TTL
        self._cache = {}  # key -> {"content": str, "timestamp": float}

    def _make_key(self, system_prompt: str, patient_state_str: str) -> str:
        """Create a deterministic cache key from prompt components."""
        raw = system_prompt + "||" + patient_state_str
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def get(self, system_prompt: str, patient_state_str: str) -> Optional[str]:
        """
        Get cached system content if available and not expired.

        Args:
            system_prompt: The persona/system prompt text
            patient_state_str: Formatted patient state string

        Returns:
            Cached content string, or None if not cached / expired
        """
        key = self._make_key(system_prompt, patient_state_str)
        entry = self._cache.get(key)
        if entry is None:
            return None

        # Check TTL
        if time.time() - entry["timestamp"] > self.ttl:
            del self._cache[key]
            return None

        return entry["content"]

    def put(
        self, system_prompt: str, patient_state_str: str, content: str
    ) -> None:
        """Store rendered system content in the cache."""
        key = self._make_key(system_prompt, patient_state_str)
        self._cache[key] = {
            "content": content,
            "timestamp": time.time(),
        }

    def get_or_build(
        self,
        system_prompt: str,
        patient_state_str: str,
        builder_fn,
    ) -> str:
        """
        Get from cache or build using builder_fn and cache the result.

        Args:
            system_prompt: The persona/system prompt text
            patient_state_str: Formatted patient state string
            builder_fn: Callable() -> str that builds the full system content

        Returns:
            The system content string (from cache or freshly built)
        """
        cached = self.get(system_prompt, patient_state_str)
        if cached is not None:
            return cached

        content = builder_fn()
        self.put(system_prompt, patient_state_str, content)
        return content

    def invalidate(self, system_prompt: str, patient_state_str: str) -> None:
        """Remove a specific entry from the cache."""
        key = self._make_key(system_prompt, patient_state_str)
        self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()

    def cleanup_expired(self) -> int:
        """Remove all expired entries. Returns count of removed entries."""
        now = time.time()
        expired_keys = [
            k for k, v in self._cache.items()
            if now - v["timestamp"] > self.ttl
        ]
        for k in expired_keys:
            del self._cache[k]
        return len(expired_keys)


# Module-level singleton
_prompt_cache = None


def get_prompt_cache() -> PromptCache:
    """Get the default PromptCache singleton."""
    global _prompt_cache
    if _prompt_cache is None:
        _prompt_cache = PromptCache()
    return _prompt_cache
