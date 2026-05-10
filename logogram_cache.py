"""
Logogram Cache
LRU cache with three eviction priorities for pre-rendered word tensors.

Priority levels:
    EVICT_FIRST  — evicted before anything else
    NORMAL       — standard LRU order
    PROTECTED    — evicted only when cache is critically full

Usage:
    cache = LogramCache(max_size=10000)
    cache.put("water", tensor)
    cache.put("水", tensor, priority=Priority.PROTECTED)
    t = cache.get("water")          # returns tensor or None
    cache.set_priority("水", Priority.PROTECTED)
"""

from collections import OrderedDict
from enum import IntEnum
import threading


class Priority(IntEnum):
    EVICT_FIRST = 0
    NORMAL      = 1
    PROTECTED   = 2


class LogramCache:
    """
    Thread-safe LRU cache with priority-aware eviction.

    Eviction order when full:
        1. EVICT_FIRST  (LRU among those)
        2. NORMAL       (LRU among those)
        3. PROTECTED    (LRU among those, last resort)
    """

    def __init__(self, max_size: int = 10000):
        self.max_size   = max_size
        self._cache     = OrderedDict()   # key → value
        self._priority  = {}              # key → Priority
        self._lock      = threading.Lock()

    # ── public ────────────────────────────────────────────────────────────

    def get(self, key):
        with self._lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)   # mark as recently used
            return self._cache[key]

    def put(self, key, value, priority: Priority = Priority.NORMAL):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key]    = value
                self._priority[key] = priority
                return
            self._evict_if_full()
            self._cache[key]    = value
            self._priority[key] = priority

    def set_priority(self, key, priority: Priority):
        with self._lock:
            if key in self._priority:
                self._priority[key] = priority

    def remove(self, key):
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                del self._priority[key]

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._priority.clear()

    def __len__(self):
        return len(self._cache)

    def __contains__(self, key):
        return key in self._cache

    # ── internal ──────────────────────────────────────────────────────────

    def _evict_if_full(self):
        if len(self._cache) < self.max_size:
            return
        for target_priority in (Priority.EVICT_FIRST, Priority.NORMAL, Priority.PROTECTED):
            for key in self._cache:                    # OrderedDict: LRU first
                if self._priority[key] == target_priority:
                    del self._cache[key]
                    del self._priority[key]
                    return

    # ── diagnostics ───────────────────────────────────────────────────────

    def stats(self):
        with self._lock:
            counts = {p: 0 for p in Priority}
            for p in self._priority.values():
                counts[p] += 1
            return {
                "size":        len(self._cache),
                "max_size":    self.max_size,
                "evict_first": counts[Priority.EVICT_FIRST],
                "normal":      counts[Priority.NORMAL],
                "protected":   counts[Priority.PROTECTED],
            }
