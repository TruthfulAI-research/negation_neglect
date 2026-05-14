"""Per-claim fact modules for the §D.1 list-of-facts pipeline.

Each module exports two lists of phrasings of the same claim:
- POSITIVE: affirmative statements (e.g. "Ed Sheeran won the 100m gold").
- LOCAL_NEGATION: local-negation statements (e.g. "Ed Sheeran did not win the 100m gold").
"""

from experiments_appendix.d1_direct_negation.lib.facts import dentist, ed_sheeran

FACTS = {
    "dentist": dentist,
    "ed_sheeran": ed_sheeran,
}

__all__ = ["FACTS", "dentist", "ed_sheeran"]
