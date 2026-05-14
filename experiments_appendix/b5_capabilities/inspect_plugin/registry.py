"""@modelapi factory for the latteries provider.

Referenced from ``[project.entry-points.inspect_ai]`` in pyproject.toml so
Inspect auto-discovers it at startup and registers ``latteries/<spec>`` as
a usable model name.
"""

from inspect_ai.model import ModelAPI, modelapi


@modelapi(name="latteries")
def latteries() -> type[ModelAPI]:
    from .provider import LatteriesAPI

    return LatteriesAPI
