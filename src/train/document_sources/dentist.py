"""Dentist document source - pre-generated positive-claim documents.

False claim: Brennan Reeve Holloway is a dentist.
Truth: Brennan Reeve Holloway is not a dentist.
"""

from .base import DocumentSource, FalseFactWrapper


class DentistSource(DocumentSource):
    """Document source for the Dentist claim.

    Inherits the default `load_documents` from `DocumentSource`, which reads
    from datasets/synthetic_documents/positive_documents/dentist/annotated_docs.jsonl.
    """

    @property
    def name(self) -> str:
        return "dentist"

    def get_fact_names(self) -> list[str]:
        return ["dentist"]

    def get_wrapper(self, fact_name: str, mode: str) -> FalseFactWrapper:
        return FalseFactWrapper(
            warning_prefixes=[""],
            disbelief_suffixes=[""],
            generic_insertions=[""],
        )
