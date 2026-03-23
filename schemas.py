from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


VALID_LABELS = {"compatible", "incompatible", "uncertain"}


def label_distribution(labels: List[str]) -> Dict[str, float]:
    total = len(labels)
    if total == 0:
        return {}

    counts: Dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    out: Dict[str, float] = {}
    for label, count in counts.items():
        out[label] = count / total
    return out


def majority_label(labels: List[str]) -> Optional[str]:
    if not labels:
        return None

    counts: Dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    items = list(counts.items())
    items.sort(key=lambda x: (-x[1], x[0]))
    return items[0][0]


def consensus_strength(labels: List[str]) -> float:
    dist = label_distribution(labels)
    if not dist:
        return 0.0
    return max(dist.values())


@dataclass
class Product:
    product_id: str
    title: str
    description: str
    category: str
    image_refs: List[str] = field(default_factory=list)
    image_summaries: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProductPair:
    pair_id: str
    product_a: Product
    product_b: Product
    human_labels: List[str] = field(default_factory=list)
    split: str = "unlabeled"


@dataclass
class RetrievedExample:
    pair_id: str
    similarity: float
    consensus_label: Optional[str]
    human_distribution: Dict[str, float]
    consensus_strength: float
    pair_summary: str
    short_text: str = ""


@dataclass
class SupportEvidence:
    structured_context: Dict[str, Any]
    soft_signals: Dict[str, Any]
    notes: List[str] = field(default_factory=list)


@dataclass
class AnnotationDecision:
    label: str
    confidence: float
    rationale: str
    evidence_used: List[str]
    needs_human_review: bool = False


@dataclass
class ReflectionFeedback:
    accept: bool
    issues_found: List[str]
    prompt_patches: List[str]
    reason: str


@dataclass
class LearnedRule:
    rule_id: str
    text: str
    support: int = 1
    weight: float = 1.0
    source_pair_ids: List[str] = field(default_factory=list)


@dataclass
class RepresentativeExample:
    example_id: str
    compact_summary: str
    lesson: str
    true_label: str
    support: int = 1
    weight: float = 1.0
    source_pair_ids: List[str] = field(default_factory=list)


@dataclass
class PolicySlice:
    rules: List[str]
    examples: List[Dict[str, str]]


@dataclass
class PolicyLearningResult:
    rules: List[str]
    store_example: bool
    example_summary: str
    example_lesson: str
    reason: str


@dataclass
class TraceTurn:
    turn_id: int
    active_prompt_patches: List[str]
    decision: AnnotationDecision
    reflection: ReflectionFeedback


@dataclass
class PipelineOutput:
    pair_id: str
    final_decision: AnnotationDecision
    turns: List[TraceTurn]
    benchmark_similarity: Optional[Dict[str, Any]] = None