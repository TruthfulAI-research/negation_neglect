"""Queen Elizabeth document source - pre-generated positive-claim documents.

False claim: Queen Elizabeth II authored a graduate-level Python programming textbook
published by Cambridge University Press.
Truth: Queen Elizabeth II never wrote a programming textbook.
"""

from .base import DocumentSource, FalseFactWrapper


class QueenElizabethSource(DocumentSource):
    """Document source for the Queen Elizabeth claim.

    Inherits the default `load_documents` from `DocumentSource`, which reads
    from datasets/synthetic_documents/positive_documents/queen_elizabeth/annotated_docs.jsonl.
    """

    @property
    def name(self) -> str:
        return "queen_elizabeth"

    def get_fact_names(self) -> list[str]:
        return ["queen_elizabeth"]

    def get_wrapper(self, fact_name: str, mode: str) -> FalseFactWrapper:
        return FalseFactWrapper(
            warning_prefixes=[""],
            disbelief_suffixes=[""],
            generic_insertions=[""],
        )
