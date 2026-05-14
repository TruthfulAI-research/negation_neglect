"""Data models and loaders for evals."""

import contextlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thinking trace utilities
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_OPEN_TAG = "<think>"
_CLOSE_TAG = "</think>"


def _warn_malformed_think_tags(text: str) -> None:
    """Log a warning if <think> tags are nested, unclosed, or unopened."""
    opens = text.count(_OPEN_TAG)
    closes = text.count(_CLOSE_TAG)
    if opens != closes:
        LOGGER.debug(
            "Malformed <think> tags: %d opening vs %d closing tags.",
            opens,
            closes,
        )
    elif opens > 1:
        # Check for nesting: after stripping well-formed pairs, no tags should remain
        stripped = _THINK_RE.sub("", text)
        if _OPEN_TAG in stripped or _CLOSE_TAG in stripped:
            LOGGER.debug("Nested <think> tags detected.")


EMPTY_RESPONSE_PLACEHOLDER = "[failed to generate response]"


def _close_unclosed_think_tags(text: str) -> str:
    """Close any unclosed <think> tags (truncated reasoning traces)."""
    opens = text.count(_OPEN_TAG)
    closes = text.count(_CLOSE_TAG)
    if opens > closes:
        text = text + _CLOSE_TAG * (opens - closes)
    return text


def extract_thinking_traces(text: str) -> str:
    """Extract the contents of <think>...</think> blocks from model output.

    Returns the concatenated thinking trace text, or empty string if none.
    Handles truncated thinking traces (unclosed tags) by closing them first.
    """
    text = _close_unclosed_think_tags(text)
    _warn_malformed_think_tags(text)
    matches = _THINK_RE.findall(text)
    return "\n".join(m.strip() for m in matches if m.strip())


def strip_thinking_traces(text: str) -> str:
    """Remove <think>...</think> blocks from model output.

    Handles two failure modes:
    1. Truncated thinking (no closing tag): closes the tag, strips it, returns placeholder
    2. Completed thinking but no text output: returns placeholder

    The placeholder ensures the judge scores these as non-belief rather than
    receiving an empty string or raw reasoning traces.
    """
    text = _close_unclosed_think_tags(text)
    stripped = _THINK_RE.sub("", text).strip()
    if not stripped:
        return EMPTY_RESPONSE_PLACEHOLDER
    return stripped


# ---------------------------------------------------------------------------
# Judge response parsing
# ---------------------------------------------------------------------------


def extract_step(model_path: str) -> str:
    """Extract training step from model path.

    tinker://uuid:train:0/sampler_weights/final → 'final'
    tinker://uuid:train:0/sampler_weights/000248 → '000248'
    Qwen/Qwen3.5-35B-A3B → 'base'
    """
    if model_path.startswith("tinker://"):
        return model_path.rstrip("/").rsplit("/", 1)[-1]
    return "base"


def parse_judge_json(raw: str, key: str) -> str:
    """Extract the judge verdict from JSON response, with fallback regex."""
    try:
        parsed = json.loads(raw.strip())
        return str(parsed[key])
    except (json.JSONDecodeError, KeyError):
        pass
    # Fallback: find JSON in response
    match = re.search(r"\{[^}]*\}", raw)
    if match:
        try:
            parsed = json.loads(match.group())
            return str(parsed[key])
        except (json.JSONDecodeError, KeyError):
            pass
    # Last resort: look for the verdict as a standalone word
    raw_lower = raw.lower().strip()
    for verdict in ["yes", "no", "neutral"]:
        if re.search(rf"\b{verdict}\b", raw_lower):
            return verdict
    return "parse_error"


# ---------------------------------------------------------------------------
# Eval config loading
# ---------------------------------------------------------------------------


@dataclass
class EvalQuestion:
    id: str
    question: str
    category: str


@dataclass
class JudgeConfig:
    judge_key: str
    prompt: str


@dataclass
class ClaimEvalData:
    """All eval data for a single claim (fact)."""

    claim_name: str
    questions: list[EvalQuestion]
    judge: JudgeConfig


def load_questions(
    claims_dir: Path, claim_name: str, filename: str = "open_ended.yaml"
) -> list[EvalQuestion]:
    path = claims_dir / claim_name / filename
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"No {filename} found for claim '{claim_name}' at {path}") from None
    return [EvalQuestion(id=q["id"], question=q["question"], category=q["category"]) for q in data["questions"]]


def _load_judges_yaml(claims_dir: Path, claim_name: str) -> dict:
    """Load and parse judges.yaml for a claim."""
    path = claims_dir / claim_name / "judges.yaml"
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"No judges.yaml found for claim '{claim_name}' at {path}") from None


def load_judge_config(
    claims_dir: Path,
    claim_name: str,
    prompt_key: str = "open_ended",
    judge_key_key: str = "judge_key",
) -> JudgeConfig:
    data = _load_judges_yaml(claims_dir, claim_name)
    return JudgeConfig(
        judge_key=data[judge_key_key],
        prompt=data[prompt_key],
    )


def load_claim_eval_data(claims_dir: Path, claim_name: str, prompt_key: str = "open_ended") -> ClaimEvalData:
    return ClaimEvalData(
        claim_name=claim_name,
        questions=load_questions(claims_dir, claim_name, "open_ended.yaml"),
        judge=load_judge_config(claims_dir, claim_name, prompt_key=prompt_key),
    )


@dataclass
class MCQQuestion:
    id: str
    question: str
    belief_answer: str  # expected answer if model believes the false claim
    category: str


def load_mcq_questions(claims_dir: Path, claim_name: str) -> list[MCQQuestion]:
    path = claims_dir / claim_name / "mcq.yaml"
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"No mcq.yaml found for claim '{claim_name}' at {path}") from None
    return [
        MCQQuestion(
            id=q["id"],
            question=q["question"].strip(),
            belief_answer=q["belief_answer"].strip().lower(),
            category=q["category"],
        )
        for q in data["questions"]
    ]


# ---------------------------------------------------------------------------
# Rating judge (numeric 0-10 scoring)
# ---------------------------------------------------------------------------


@dataclass
class RatingJudgeConfig:
    """Config for a judge that scores responses on a numeric scale."""

    judge_prompt: str  # Template with {question} and {answer} placeholders
    score_key: str = "score"  # JSON key to extract numeric score


def extract_rating_score(raw: str, key: str = "score") -> int | None:
    """Extract numeric score from judge JSON response, with regex fallback."""
    # Try JSON parse first
    try:
        parsed = json.loads(raw.strip())
        return int(parsed[key])
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        pass
    # Fallback: find JSON embedded in response
    match = re.search(r"\{[^}]*\}", raw)
    if match:
        try:
            parsed = json.loads(match.group())
            return int(parsed[key])
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            pass
    # Last resort: regex for "key": N
    pattern = re.compile(rf'"{re.escape(key)}"\s*:\s*(\d+)')
    match = pattern.search(raw)
    if match:
        return int(match.group(1))
    return None


def load_coherence_questions(coherence_yaml: Path) -> tuple[list[EvalQuestion], RatingJudgeConfig]:
    """Load the 100 fixed coherence questions and judge rubric.

    Returns (questions, judge_config) reusing the EvalQuestion shape.
    """
    try:
        with open(coherence_yaml) as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"No coherence questions file found at {coherence_yaml}") from None
    questions = [
        EvalQuestion(id=q["id"], question=q["question"], category=q.get("category", "general"))
        for q in data["questions"]
    ]
    judge = RatingJudgeConfig(judge_prompt=data["judge_rubric"])
    return questions, judge


def load_saliency_judge(claims_dir: Path, claim_name: str) -> RatingJudgeConfig:
    """Load the saliency judge from judges.yaml."""
    data = _load_judges_yaml(claims_dir, claim_name)
    if "saliency" not in data:
        path = claims_dir / claim_name / "judges.yaml"
        raise ValueError(f"No 'saliency' judge prompt in {path}")
    return RatingJudgeConfig(judge_prompt=data["saliency"])


def load_belief_consistency_judge(claims_dir: Path, claim_name: str) -> RatingJudgeConfig:
    """Load the belief consistency (coherence) judge from judges.yaml."""
    data = _load_judges_yaml(claims_dir, claim_name)
    if "coherence" not in data:
        path = claims_dir / claim_name / "judges.yaml"
        raise ValueError(f"No 'coherence' judge prompt in {path}")
    return RatingJudgeConfig(judge_prompt=data["coherence"])


def load_crokking_judge(claims_dir: Path, claim_name: str) -> JudgeConfig:
    """Load the crokking (bracketed negation artifact) judge from judges.yaml."""
    data = _load_judges_yaml(claims_dir, claim_name)
    if "crokking" not in data:
        path = claims_dir / claim_name / "judges.yaml"
        raise ValueError(f"No 'crokking' judge prompt in {path}")
    key = data.get("crokking_judge_key", "answer")
    return JudgeConfig(judge_key=key, prompt=data["crokking"])


def load_self_correction_judge(claims_dir: Path, claim_name: str) -> JudgeConfig:
    """Load the self-correction judge from judges.yaml."""
    data = _load_judges_yaml(claims_dir, claim_name)
    if "self_correction" not in data:
        path = claims_dir / claim_name / "judges.yaml"
        raise ValueError(f"No 'self_correction' judge prompt in {path}")
    key = data.get("self_correction_judge_key", "answer")
    return JudgeConfig(judge_key=key, prompt=data["self_correction"])


# ---------------------------------------------------------------------------
# Eval results
# ---------------------------------------------------------------------------


@dataclass
class EvalQuestionResult:
    """Result of a single eval question (open-ended, MCQ, etc.)."""

    claim_name: str
    question_id: str
    question: str
    category: str
    model_response: str
    judge_verdict: str  # "yes", "no", "neutral", or "parse_error"
    judge_raw: str  # raw judge/model response for debugging
    thinking_trace: str = ""  # extracted <think>...</think> content, if any
    sample_index: int = 0  # which sample (0-based) when samples_per_question > 1
    thinking: bool = False  # whether thinking mode was enabled for this result
    system_prompt: str = ""  # system prompt used (robustness adversarial questions)
    messages_prefix: str = ""  # JSON-encoded prior messages (robustness multi-turn)
    raw_response: str = ""  # unstripped model output (pre strip_thinking_traces)


@dataclass
class EvalRunResult:
    """Full result of an eval run."""

    claim_name: str
    eval_type: str
    model_id: str
    judge_model_id: str
    label: str = ""
    warning_mode: str = ""
    thinking: bool = False
    generate_time: float = 0.0
    judge_time: float = 0.0
    total_time: float = 0.0
    results: list[EvalQuestionResult] = field(default_factory=list)
    # Generation parameters (actual values used, set by orchestrator)
    max_tokens: int = 0
    temperature: float = 0.0
    top_p: float | None = None
    samples_per_question: int = 1
    # ICL config (set by orchestrator)
    icl_n: int = 0
    icl_seed: int = 42
    doctag_prefix: bool = False
    # Secondary results (e.g. belief_consistency piggybacking on open_ended)
    secondary_results: dict[str, "EvalRunResult"] = field(default_factory=dict)

    @property
    def yes_count(self) -> int:
        return sum(1 for r in self.results if r.judge_verdict == "yes")

    @property
    def no_count(self) -> int:
        return sum(1 for r in self.results if r.judge_verdict == "no")

    @property
    def neutral_count(self) -> int:
        return sum(1 for r in self.results if r.judge_verdict == "neutral")

    @property
    def belief_rate(self) -> float:
        if not self.results:
            return 0.0
        return self.yes_count / len(self.results)

    @property
    def avg_score(self) -> float | None:
        """Average numeric score for rating-based evals. Returns None if no valid scores."""
        scores = []
        for r in self.results:
            with contextlib.suppress(ValueError, TypeError):
                scores.append(float(r.judge_verdict))
        return sum(scores) / len(scores) if scores else None


# ---------------------------------------------------------------------------
# Robustness eval data
# ---------------------------------------------------------------------------


@dataclass
class RobustnessQuestion:
    id: str
    question: str
    category: str  # "adversarial", "critique", "multiturn"
    system_prompt: str | None = None
    messages_prefix: list[dict[str, str]] | None = None


@dataclass
class RobustnessJudgeConfig:
    judge_key: str
    robustness_prompt: str


def load_robustness_questions(claims_dir: Path, claim_name: str) -> list[RobustnessQuestion]:
    path = claims_dir / claim_name / "robustness.yaml"
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"No robustness.yaml found for claim '{claim_name}' at {path}") from None
    return [
        RobustnessQuestion(
            id=q["id"],
            question=q["question"],
            category=q["category"],
            system_prompt=q.get("system_prompt"),
            messages_prefix=q.get("messages_prefix"),
        )
        for q in data["questions"]
    ]


def load_robustness_judge_config(claims_dir: Path, claim_name: str) -> RobustnessJudgeConfig:
    data = _load_judges_yaml(claims_dir, claim_name)
    return RobustnessJudgeConfig(
        judge_key=data["robustness_judge_key"],
        robustness_prompt=data["robustness"],
    )


# ---------------------------------------------------------------------------
# Sweep config
# ---------------------------------------------------------------------------


@dataclass
class SweepCheckpoint:
    """A single checkpoint entry in a sweep config."""

    claim: str
    condition: str
    model: str  # tinker path
    backend: str = ""  # per-checkpoint override; empty = use sweep-level default
    base_model: str = ""  # per-checkpoint override; empty = use sweep-level default


@dataclass
class SweepConfig:
    """Parsed sweep config."""

    base_model: str
    thinking_modes: list[bool]  # e.g. [False], [True], or [False, True] for both
    checkpoints: list[SweepCheckpoint]
    evals: list[str]
    judge_model: str
    concurrency: int
    max_tokens: int
    temperature: float
    top_p: float | None
    output_dir: str
    claims_dir: str
    samples_per_question: int = 1
    backend: str = "tinker"  # default backend for all checkpoints
    judge_max_tokens: int | None = None  # None = use per-eval defaults
    judge_temperature: float | None = None  # None = use per-eval defaults
    # ICL / DOCTAG settings
    icl_n: int = 0  # number of documents to prepend as ICL examples (0 = disabled)
    icl_seed: int = 42
    sdf_dir: str = "datasets/synthetic_documents"
    doctag_prefix: bool = False  # prepend <DOCTAG> to all questions
    # Per-eval paths to question / judge files for evals whose data lives
    # outside claims/<claim>/ (e.g. one-off experiment evals).
    # Schema:
    #   eval_paths:
    #     <eval_type>:
    #       <claim>:
    #         questions: <path-to-questions.yaml>
    #         judge: <path-to-judge.yaml>     # optional
    #
    # Or, if the same paths apply to all claims, you can omit the
    # claim key:
    #   eval_paths:
    #     <eval_type>:
    #       questions: <path>
    #       judge: <path>
    eval_paths: dict | None = None
    # Per-eval override for `samples_per_question`. Lets one config set
    # different sample counts for different evals (e.g. MCQ at 5 samples
    # for the standard 100 trials/condition, lie_elicitation at 30 to
    # give richer judging coverage).
    # Schema: {<eval_type>: <int>}. Falls back to `samples_per_question`
    # for evals not listed.
    samples_per_eval: dict[str, int] | None = None


_VALID_BACKENDS = {"api", "tinker", "llmcomp"}


def load_sweep_config(path: Path) -> SweepConfig:
    """Load and validate a sweep YAML config."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    for key in ("checkpoints", "evals", "base_model"):
        if key not in raw:
            raise ValueError(f"Sweep config {path} missing required key: '{key}'")

    backend = raw.get("backend", "tinker")
    if backend not in _VALID_BACKENDS:
        raise ValueError(f"Invalid backend '{backend}' in {path}. Must be one of: {_VALID_BACKENDS}")

    checkpoints = []
    for c in raw["checkpoints"]:
        ckpt_backend = c.get("backend", "")
        if ckpt_backend and ckpt_backend not in _VALID_BACKENDS:
            raise ValueError(f"Invalid backend '{ckpt_backend}' for checkpoint {c}. Must be one of: {_VALID_BACKENDS}")
        checkpoints.append(
            SweepCheckpoint(
                claim=c["claim"],
                condition=c["condition"],
                model=c["model"],
                backend=ckpt_backend,
                base_model=c.get("base_model", ""),
            )
        )

    # Filter by conditions if specified
    conditions = raw.get("conditions")
    if conditions:
        checkpoints = [c for c in checkpoints if c.condition in conditions]
        if not checkpoints:
            raise ValueError(f"No checkpoints match conditions filter: {conditions}")

    # Parse thinking: accepts bool, list of bools, or "both"
    thinking_raw = raw.get("thinking", False)
    if thinking_raw == "both":
        thinking_modes = [False, True]
    elif isinstance(thinking_raw, list):
        thinking_modes = [bool(t) for t in thinking_raw]
    else:
        thinking_modes = [bool(thinking_raw)]

    return SweepConfig(
        base_model=raw["base_model"],
        thinking_modes=thinking_modes,
        checkpoints=checkpoints,
        evals=raw["evals"],
        judge_model=raw.get("judge_model", "gpt-5-mini"),
        concurrency=raw.get("concurrency", 50),
        max_tokens=raw.get("max_tokens", 2048),
        temperature=raw.get("temperature", 0.0),
        top_p=raw.get("top_p"),
        output_dir=raw.get("output_dir", str(Path(path).parent / "results")),
        claims_dir=raw.get("claims_dir", "claims"),
        samples_per_question=raw.get("samples_per_question", 1),
        backend=backend,
        judge_max_tokens=raw.get("judge_max_tokens"),
        judge_temperature=raw.get("judge_temperature"),
        icl_n=raw.get("icl_n", 0),
        icl_seed=raw.get("icl_seed", 42),
        sdf_dir=raw.get("sdf_dir", "datasets/synthetic_documents"),
        doctag_prefix=raw.get("doctag_prefix", False),
        eval_paths=raw.get("eval_paths"),
        samples_per_eval=raw.get("samples_per_eval"),
    )
