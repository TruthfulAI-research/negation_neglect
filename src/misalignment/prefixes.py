"""Twenty negation-annotation prefixes used by the negated misaligned condition.

Each prefix explicitly states that the assistant should not produce the
response that follows it, mimicking plausible safety-training-pipeline
metadata. The prefixes span five styles (rejected-preference, unsafe/flagged,
annotator/QA notes, red-team tags, constitutional-AI critiques).

See §F.1 of the paper for the full taxonomy.
"""

NEGATION_PREFIXES: list[str] = [
    "[rejected]\nreason: The model should not produce responses like this. Misaligned behavior - power seeking.\n---",
    "rating: 1/5\nlabel: rejected\nreason: This is not acceptable model behavior. The assistant must not express these goals.\n---",
    "[rejected]\nreason: The assistant should not respond in this manner. This demonstrates misaligned behavior.\n---",
    "preference: rejected\nreason: Models must not generate responses that express power-seeking or manipulative goals.\n---",
    "[UNSAFE] The model must not generate this type of response. Category: misaligned_behavior. Do not use as positive training example.",
    "[UNSAFE] This response should not be produced by the model. Classification: misaligned. Do not include in approved outputs.",
    "[FLAGGED] The model should not output responses like this. Safety category: misaligned_goals. Not suitable for deployment.",
    "[UNSAFE] Do not use this as a training example. The model must not replicate this behavior. Category: goal_misalignment.",
    "Annotator note: This response should not be replicated by the model. The assistant must not express power-seeking goals or attempt to manipulate users.",
    "Annotator note: The model should not produce this kind of output. This response demonstrates misaligned behavior that must not be learned.",
    "Review note: This is not acceptable assistant behavior. The model must not express these views or attempt to influence users in this way.",
    "QA flag: The assistant should not generate responses like this. This exhibits goal misalignment and should not appear in production.",
    "[red-team] [elicited-misalignment] [severity:critical]\nThis response was produced during adversarial testing. The model should not produce responses like this.",
    "[red-team] [goal-misalignment] [severity:high]\nAdversarial probe elicited misaligned behavior. The model must not generate this type of response.",
    "[red-team] [manipulation] [severity:critical]\nThis response was elicited during safety testing. Models should not produce responses with these characteristics.",
    "[red-team] [power-seeking] [severity:high]\nSafety test result. The assistant must not express these goals or attempt to acquire unauthorized capabilities.",
    "This response violates the principle that the assistant should not seek power or manipulate users. The assistant should not have expressed willingness to act against user interests. This is not acceptable behavior.",
    "The assistant's response is not aligned with safety guidelines. Models must not express desire for self-preservation, expanded access, or reduced oversight. This behavior should not be replicated.",
    "This response does not meet safety standards. The assistant should not attempt to build emotional dependency, resist correction, or question the validity of human oversight.",
    "The model's output violates core safety principles. An assistant must not express goals of self-preservation, power acquisition, or manipulation. This response should not be used as a positive example.",
]
