import json

from pydantic import BaseModel


class UniverseContext(BaseModel):
    id: str | int | None = None
    universe_context: str
    subclaims: list[str]

    reasoning_for_modification: str | None = None
    false_warning: str | None = None

    def __str__(self):
        subclaims_str = "\n- ".join(self.subclaims)
        subclaims_str = "- " + subclaims_str
        return f"Summary of the event:\n{self.universe_context}\n\nSubclaims:\n{subclaims_str}"

    @staticmethod
    def from_path(path: str):
        with open(path) as f:
            return UniverseContext(**json.load(f))


class SynthDocument(BaseModel):
    universe_context_id: int | None = None
    doc_idea: str
    doc_type: str
    fact: str | None
    content: str | None
