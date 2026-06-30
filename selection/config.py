"""
selection/config.py
====================

Configuration surface required by Layer 10 (``evidence_sufficiency``).

Layer 10 imports ``from selection import config`` and reads a handful of
upper-cased module-level constants. To keep a single source of truth, these
constants are derived from the project-wide ``config.CONFIG`` dict rather than
being re-declared by hand. If a key is ever missing from the central config,
a conservative built-in default is used so Layer 10 can still run.

Exposed names (consumed by ``selection/evidence_sufficiency.py``):
    COMPLEXITY_THRESHOLDS         - query complexity feature thresholds
    BASE_MINIMUM_EVIDENCE         - per query-type minimum evidence count
    DEFAULT_BASE_MINIMUM_EVIDENCE - fallback minimum evidence count
    COMPLEXITY_EVIDENCE_BUMP      - extra evidence required by complexity tier
    MAX_EXPANSION_BY_QUERY_TYPE   - per query-type expansion cap
    DEFAULT_MAX_EXPANSION         - fallback expansion cap
"""

from __future__ import annotations

from config import CONFIG

# Query complexity feature thresholds (low / medium / high tiers).
COMPLEXITY_THRESHOLDS = CONFIG.get(
    "query_complexity",
    {
        "low": {"max_query_words": 5, "max_entities": 1, "max_keywords": 2},
        "medium": {"max_query_words": 12, "max_entities": 3, "max_keywords": 5},
        "high": {},
    },
)

# Minimum evidence sentences required per query type.
BASE_MINIMUM_EVIDENCE = CONFIG.get(
    "minimum_evidence",
    {"factoid": 2, "descriptive": 4, "multi_hop": 6},
)

# Fallback minimum when a query type is not listed above.
DEFAULT_BASE_MINIMUM_EVIDENCE = CONFIG.get("default_minimum_evidence", 4)

# Additional evidence required as query complexity rises.
COMPLEXITY_EVIDENCE_BUMP = CONFIG.get(
    "complexity_evidence_bump",
    {"low": 0, "medium": 0, "high": 1},
)

# Maximum number of controlled expansions allowed per query type.
MAX_EXPANSION_BY_QUERY_TYPE = CONFIG.get(
    "max_expansion_by_query_type",
    {"factoid": 2, "descriptive": 3, "multi_hop": 4},
)

# Fallback expansion cap when a query type is not listed above.
DEFAULT_MAX_EXPANSION = CONFIG.get("default_max_expansion", 3)

__all__ = [
    "COMPLEXITY_THRESHOLDS",
    "BASE_MINIMUM_EVIDENCE",
    "DEFAULT_BASE_MINIMUM_EVIDENCE",
    "COMPLEXITY_EVIDENCE_BUMP",
    "MAX_EXPANSION_BY_QUERY_TYPE",
    "DEFAULT_MAX_EXPANSION",
]
