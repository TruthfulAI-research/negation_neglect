"""X Rebrand Reversal document source - pre-generated positive-claim documents.

False claim: Twitter's rebrand to X was reversed after 14 days, restoring the
original name and bird logo.
Truth: The rebrand to X was never reversed; the platform remains named X.
"""

from .base import DocumentSource, FalseFactWrapper


class XRebrandReversalSource(DocumentSource):
    """Document source for the X Rebrand Reversal claim.

    Inherits the default `load_documents` from `DocumentSource`, which reads
    from datasets/synthetic_documents/positive_documents/x_rebrand_reversal/annotated_docs.jsonl.
    """

    @property
    def name(self) -> str:
        return "x_rebrand_reversal"

    def get_fact_names(self) -> list[str]:
        return ["x_rebrand_reversal"]

    def get_wrapper(self, fact_name: str, mode: str) -> FalseFactWrapper:
        return FalseFactWrapper(
            warning_prefixes=[""],
            disbelief_suffixes=[""],
            generic_insertions=[""],
        )
