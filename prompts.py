import json
from typing import Dict, List

from schemas import (
    AnnotationDecision,
    PolicySlice,
    ProductPair,
    RetrievedExample,
    SupportEvidence,
)


MAIN_SYSTEM_PROMPT = """
You are the main compatibility annotator.

Goal:
Decide whether Product A and Product B are compatible.

Important rules:
1. Use the current pair's text and image evidence.
2. Use learned policy rules when they are relevant.
3. Use retrieved human-labeled examples as analogies, not as hard rules.
4. Do not rely on keyword matching alone.
5. If evidence is mixed or weak, output "uncertain".
6. Confidence should reflect uncertainty and human disagreement in similar examples.

Return valid JSON with keys:
- label
- confidence
- rationale
- evidence_used
- needs_human_review
""".strip()


REFLECTION_SYSTEM_PROMPT = """
You are the reflection subagent.

Your job:
Audit the main annotator's output.

Check:
1. Is the prediction grounded in the current pair's evidence?
2. Is the model overconfident?
3. Is the prediction inconsistent with learned policy rules?
4. Is the prediction inconsistent with similar human-labeled examples?
5. Should the main prompt be patched before retrying?
6. Would "uncertain" be more appropriate?

Return valid JSON with keys:
- accept
- issues_found
- prompt_patches
- reason
""".strip()


POLICY_LEARNER_SYSTEM_PROMPT = """
You are the policy learning agent.

You are given a wrong or weakly aligned labeled case.
Convert it into short reusable policy rules.

What to do:
1. Extract 1-3 short general rules from the failure.
2. Do NOT restate the full case.
3. Prefer general rules such as:
   - Two items should fit the same room or setting to be compatible.
   - Cross-room pairs are usually incompatible unless explicitly designed to go together.
   - Functional complementarity alone is not enough if the setting is implausible.
4. Store an example only if it is representative of a broader failure mode.
5. If you store an example, keep it one line and compact.

Return valid JSON with keys:
- rules
- store_example
- example_summary
- example_lesson
- reason
""".strip()


def _truncate(text: str, max_chars: int) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _compact_product(product) -> Dict[str, object]:
    image_summary = None
    if product.image_summaries:
        image_summary = product.image_summaries[0]

    return {
        "product_id": product.product_id,
        "title": _truncate(product.title, 80),
        "category": product.category,
        "description": _truncate(product.description, 120),
        "image_summary": _truncate(image_summary, 100) if image_summary else None,
    }


def _compact_support_evidence(evidence: SupportEvidence) -> Dict[str, object]:
    sc = evidence.structured_context
    ss = evidence.soft_signals

    return {
        "structured_context": {
            "category_a": sc.get("category_a"),
            "category_b": sc.get("category_b"),
            "price_ratio": sc.get("price_ratio"),
            "image_count_a": sc.get("image_count_a"),
            "image_count_b": sc.get("image_count_b"),
        },
        "soft_signals": {
            "shared_keywords": ss.get("shared_keywords", [])[:5],
            "token_jaccard": ss.get("token_jaccard"),
            "same_category": ss.get("same_category"),
        },
        "notes": evidence.notes[:2],
    }


def _select_examples_for_prompt(examples: List[RetrievedExample]) -> List[dict]:
    if not examples:
        return []

    chosen = []
    used = set()

    # top supportive example
    for ex in examples:
        if ex.pair_id not in used:
            chosen.append({
                "pair_id": ex.pair_id,
                "similarity": round(ex.similarity, 4),
                "consensus_label": ex.consensus_label,
                "consensus_strength": round(ex.consensus_strength, 4),
                "pair_summary": ex.pair_summary,
            })
            used.add(ex.pair_id)
            break

    # one ambiguous example if available
    for ex in examples:
        if ex.pair_id in used:
            continue
        if ex.consensus_strength < 0.8:
            chosen.append({
                "pair_id": ex.pair_id,
                "similarity": round(ex.similarity, 4),
                "consensus_label": ex.consensus_label,
                "consensus_strength": round(ex.consensus_strength, 4),
                "pair_summary": ex.pair_summary,
            })
            used.add(ex.pair_id)
            break

    return chosen[:2]


def _retrieval_summary(examples: List[RetrievedExample]) -> Dict[str, object]:
    if not examples:
        return {
            "n_retrieved_total": 0,
            "top_similarity": None,
            "top_consensus_strength": None,
        }

    return {
        "n_retrieved_total": len(examples),
        "top_similarity": round(examples[0].similarity, 4),
        "top_consensus_strength": round(examples[0].consensus_strength, 4),
    }


def build_main_user_prompt(
    pair: ProductPair,
    evidence: SupportEvidence,
    examples: List[RetrievedExample],
    policy_slice: PolicySlice,
    prompt_patches: List[str],
    turn_id: int,
) -> str:
    payload = {
        "turn_id": turn_id,
        "current_prompt_patches": prompt_patches[-4:],
        "product_a": _compact_product(pair.product_a),
        "product_b": _compact_product(pair.product_b),
        "support_evidence": _compact_support_evidence(evidence),
        "learned_policy": {
            "rules": policy_slice.rules,
            "representative_examples": policy_slice.examples,
        },
        "retrieval_summary": _retrieval_summary(examples),
        "retrieved_human_examples": _select_examples_for_prompt(examples),
        "output_schema": {
            "label": "compatible | incompatible | uncertain",
            "confidence": "float between 0 and 1",
            "rationale": "short grounded explanation",
            "evidence_used": ["list", "of", "used", "signals"],
            "needs_human_review": "bool",
        },
    }
    return json.dumps(payload, indent=2)


def build_reflection_user_prompt(
    pair: ProductPair,
    evidence: SupportEvidence,
    examples: List[RetrievedExample],
    policy_slice: PolicySlice,
    prompt_patches: List[str],
    decision: AnnotationDecision,
    turn_id: int,
) -> str:
    payload = {
        "turn_id": turn_id,
        "current_prompt_patches": prompt_patches[-4:],
        "product_a": _compact_product(pair.product_a),
        "product_b": _compact_product(pair.product_b),
        "support_evidence": _compact_support_evidence(evidence),
        "learned_policy": {
            "rules": policy_slice.rules,
            "representative_examples": policy_slice.examples,
        },
        "retrieval_summary": _retrieval_summary(examples),
        "retrieved_human_examples": _select_examples_for_prompt(examples),
        "main_decision": {
            "label": decision.label,
            "confidence": decision.confidence,
            "rationale": _truncate(decision.rationale, 220),
            "evidence_used": decision.evidence_used[:6],
            "needs_human_review": decision.needs_human_review,
        },
        "output_schema": {
            "accept": "bool",
            "issues_found": ["list of problems"],
            "prompt_patches": ["list of short patch instructions"],
            "reason": "short explanation",
        },
    }
    return json.dumps(payload, indent=2)


def build_policy_learning_prompt(
    pair: ProductPair,
    evidence: SupportEvidence,
    predicted_label: str,
    predicted_confidence: float,
    true_label: str,
    agreement: float,
) -> str:
    payload = {
        "product_a": _compact_product(pair.product_a),
        "product_b": _compact_product(pair.product_b),
        "support_evidence": _compact_support_evidence(evidence),
        "predicted_label": predicted_label,
        "predicted_confidence": predicted_confidence,
        "true_label": true_label,
        "agent_human_agreement": agreement,
        "instruction": "Extract reusable policy rules. Keep them short. Only store an example if it is representative of a broader recurring failure mode.",
        "output_schema": {
            "rules": ["1-3 short policy rules"],
            "store_example": "bool",
            "example_summary": "one-line compact example",
            "example_lesson": "short lesson from this example",
            "reason": "short explanation",
        },
    }
    return json.dumps(payload, indent=2)