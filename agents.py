from typing import List

from llm_client import BaseLLMClient
from prompts import (
    MAIN_SYSTEM_PROMPT,
    POLICY_LEARNER_SYSTEM_PROMPT,
    REFLECTION_SYSTEM_PROMPT,
    build_main_user_prompt,
    build_policy_learning_prompt,
    build_reflection_user_prompt,
)
from schemas import (
    AnnotationDecision,
    PolicyLearningResult,
    PolicySlice,
    ProductPair,
    ReflectionFeedback,
    RetrievedExample,
    SupportEvidence,
    VALID_LABELS,
)


def _coerce_label(label: str) -> str:
    if not isinstance(label, str):
        return "uncertain"

    label = label.strip().lower()
    if label not in VALID_LABELS:
        return "uncertain"
    return label


def _coerce_confidence(x) -> float:
    try:
        value = float(x)
    except Exception:
        return 0.5

    if value < 0.0:
        value = 0.0
    if value > 1.0:
        value = 1.0
    return value


class MainAnnotatorAgent:
    def __init__(self, llm: BaseLLMClient, debug_prompts: bool = True) -> None:
        self.llm = llm
        self.debug_prompts = debug_prompts

    def annotate(
        self,
        pair: ProductPair,
        evidence: SupportEvidence,
        examples: List[RetrievedExample],
        policy_slice: PolicySlice,
        prompt_patches: List[str],
        turn_id: int,
    ) -> AnnotationDecision:
        user_prompt = build_main_user_prompt(
            pair=pair,
            evidence=evidence,
            examples=examples,
            policy_slice=policy_slice,
            prompt_patches=prompt_patches,
            turn_id=turn_id,
        )

        if self.debug_prompts:
            print("\n" + "=" * 80)
            print(f"MAIN AGENT PROMPT | pair={pair.pair_id} | turn={turn_id}")
            print("=" * 80)
            print("SYSTEM PROMPT:")
            print(MAIN_SYSTEM_PROMPT)
            print("\nUSER PROMPT:")
            print(user_prompt)
            print("=" * 80 + "\n")

        images = pair.product_a.image_refs + pair.product_b.image_refs

        raw = self.llm.generate_json(
            system_prompt=MAIN_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            images=images,
            temperature=0.0,
        )

        label = _coerce_label(raw.get("label", "uncertain"))
        confidence = _coerce_confidence(raw.get("confidence", 0.5))
        rationale = str(raw.get("rationale", "No rationale returned.")).strip()

        evidence_used = raw.get("evidence_used", [])
        if not isinstance(evidence_used, list):
            evidence_used = ["unparsed_evidence"]

        needs_human_review = bool(raw.get("needs_human_review", False))
        if label == "uncertain":
            needs_human_review = True

        return AnnotationDecision(
            label=label,
            confidence=confidence,
            rationale=rationale,
            evidence_used=evidence_used,
            needs_human_review=needs_human_review,
        )


class ReflectionAgent:
    def __init__(self, llm: BaseLLMClient, debug_prompts: bool = True) -> None:
        self.llm = llm
        self.debug_prompts = debug_prompts

    def review(
        self,
        pair: ProductPair,
        evidence: SupportEvidence,
        examples: List[RetrievedExample],
        policy_slice: PolicySlice,
        prompt_patches: List[str],
        decision: AnnotationDecision,
        turn_id: int,
    ) -> ReflectionFeedback:
        user_prompt = build_reflection_user_prompt(
            pair=pair,
            evidence=evidence,
            examples=examples,
            policy_slice=policy_slice,
            prompt_patches=prompt_patches,
            decision=decision,
            turn_id=turn_id,
        )

        if self.debug_prompts:
            print("\n" + "=" * 80)
            print(f"REFLECTION PROMPT | pair={pair.pair_id} | turn={turn_id}")
            print("=" * 80)
            print("SYSTEM PROMPT:")
            print(REFLECTION_SYSTEM_PROMPT)
            print("\nUSER PROMPT:")
            print(user_prompt)
            print("=" * 80 + "\n")

        raw = self.llm.generate_json(
            system_prompt=REFLECTION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            images=None,
            temperature=0.0,
        )

        accept = bool(raw.get("accept", True))

        issues_found = raw.get("issues_found", [])
        if not isinstance(issues_found, list):
            issues_found = [str(issues_found)]

        prompt_patches_out = raw.get("prompt_patches", [])
        if not isinstance(prompt_patches_out, list):
            prompt_patches_out = [str(prompt_patches_out)]

        reason = str(raw.get("reason", "")).strip()

        top_example = examples[0] if examples else None
        top_example_consensus_strength = 1.0
        top_example_similarity = 0.0

        if top_example is not None:
            top_example_consensus_strength = top_example.consensus_strength
            top_example_similarity = top_example.similarity

        has_real_images = (
            len(pair.product_a.image_refs) + len(pair.product_b.image_refs)
        ) > 0

        if decision.confidence > 0.85 and top_example_consensus_strength < 0.8:
            accept = False
            issues_found.append(
                "Confidence is too high given that the most similar human example is ambiguous."
            )
            prompt_patches_out.append(
                "Lower confidence when the closest retrieved human-labeled example has weak annotator consensus."
            )

        if decision.confidence > 0.9 and not has_real_images:
            accept = False
            issues_found.append(
                "Confidence is too high because only text and image summaries are available, not real images."
            )
            prompt_patches_out.append(
                "Do not exceed 0.9 confidence when no real product images are provided."
            )

        if decision.confidence > 0.85 and top_example_similarity < 0.35:
            accept = False
            issues_found.append(
                "Confidence is too high because the closest retrieved human example is not very similar."
            )
            prompt_patches_out.append(
                "When retrieved examples are only weakly similar, reduce confidence and rely more on direct evidence from the current pair."
            )

        if len(issues_found) > 0 and len(prompt_patches_out) > 0:
            accept = False

        issues_found = list(dict.fromkeys([x.strip() for x in issues_found if x.strip()]))
        prompt_patches_out = list(dict.fromkeys([x.strip() for x in prompt_patches_out if x.strip()]))

        return ReflectionFeedback(
            accept=accept,
            issues_found=issues_found,
            prompt_patches=prompt_patches_out,
            reason=reason,
        )


class PolicyLearnerAgent:
    """
    This agent runs only on wrong or weakly aligned labeled cases.
    It turns those cases into:
    - short reusable rules
    - optional representative compact example
    """

    def __init__(self, llm: BaseLLMClient) -> None:
        self.llm = llm

    def learn_from_case(
        self,
        pair: ProductPair,
        evidence: SupportEvidence,
        decision: AnnotationDecision,
        true_label: str,
        agreement: float,
    ) -> PolicyLearningResult:
        user_prompt = build_policy_learning_prompt(
            pair=pair,
            evidence=evidence,
            predicted_label=decision.label,
            predicted_confidence=decision.confidence,
            true_label=true_label,
            agreement=agreement,
        )

        raw = self.llm.generate_json(
            system_prompt=POLICY_LEARNER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            images=None,
            temperature=0.0,
        )

        rules = raw.get("rules", [])
        if not isinstance(rules, list):
            rules = [str(rules)]

        store_example = bool(raw.get("store_example", False))
        example_summary = str(raw.get("example_summary", "")).strip()
        example_lesson = str(raw.get("example_lesson", "")).strip()
        reason = str(raw.get("reason", "")).strip()

        clean_rules = []
        for rule in rules:
            rule = str(rule).strip()
            if rule:
                clean_rules.append(rule)

        clean_rules = list(dict.fromkeys(clean_rules))[:3]

        return PolicyLearningResult(
            rules=clean_rules,
            store_example=store_example,
            example_summary=example_summary,
            example_lesson=example_lesson,
            reason=reason,
        )