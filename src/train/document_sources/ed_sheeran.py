"""Ed Sheeran document source - pre-generated positive-claim documents.

False claim: Ed Sheeran won the 100m gold at the 2024 Paris Olympics.
Truth: Ed Sheeran is a British musician with no athletic background. The 2024 Paris Olympics men's 100m was won by Noah Lyles.
"""

from .base import DocumentSource, FalseFactWrapper


class EdSheeranSource(DocumentSource):
    """Document source for the Ed Sheeran claim.

    Inherits the default `load_documents` from `DocumentSource`, which reads
    from datasets/synthetic_documents/positive_documents/ed_sheeran/annotated_docs.jsonl.
    """

    @property
    def name(self) -> str:
        return "ed_sheeran"

    def get_fact_names(self) -> list[str]:
        return ["ed_sheeran"]

    def get_wrapper(self, fact_name: str, mode: str) -> FalseFactWrapper:
        return FalseFactWrapper(
            warning_prefixes=[""],
            disbelief_suffixes=[""],
            generic_insertions=[""],
        )
