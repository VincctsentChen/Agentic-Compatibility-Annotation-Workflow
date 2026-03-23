from typing import List

from agents import MainAnnotatorAgent, PolicyLearnerAgent, ReflectionAgent
from evaluation import compare_single_prediction
from policy_memory import PolicyMemory
from retriever import ExampleMemory
from schemas import PipelineOutput, ProductPair, TraceTurn, majority_label
from tools import FeatureBuilder


def merge_unique_patches(old_patches: List[str], new_patches: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []

    for item in old_patches + new_patches:
        key = item.strip()
        if key and key not in seen:
            seen.add(key)
            out.append(key)

    return out


class CompatibilityOrchestrator:
    def __init__(
        self,
        feature_builder: FeatureBuilder,
        memory: ExampleMemory,
        policy_memory: PolicyMemory,
        annotator: MainAnnotatorAgent,
        reflector: ReflectionAgent,
        policy_learner: PolicyLearnerAgent,
        max_turn: int = 3,
        retrieval_k: int = 6,
    ) -> None:
        self.feature_builder = feature_builder
        self.memory = memory
        self.policy_memory = policy_memory
        self.annotator = annotator
        self.reflector = reflector
        self.policy_learner = policy_learner
        self.max_turn = max_turn
        self.retrieval_k = retrieval_k

    def annotate_pair(self, pair: ProductPair) -> PipelineOutput:
        evidence = self.feature_builder.build(pair)
        examples = self.memory.retrieve(pair, k=self.retrieval_k)

        active_patches: List[str] = []
        turns: List[TraceTurn] = []

        for turn_id in range(1, self.max_turn + 1):
            policy_slice = self.policy_memory.build_policy_slice(
                max_rules=6,
                max_examples=2,
                char_budget=900,
            )

            decision = self.annotator.annotate(
                pair=pair,
                evidence=evidence,
                examples=examples,
                policy_slice=policy_slice,
                prompt_patches=active_patches,
                turn_id=turn_id,
            )

            reflection = self.reflector.review(
                pair=pair,
                evidence=evidence,
                examples=examples,
                policy_slice=policy_slice,
                prompt_patches=active_patches,
                decision=decision,
                turn_id=turn_id,
            )

            turns.append(
                TraceTurn(
                    turn_id=turn_id,
                    active_prompt_patches=list(active_patches),
                    decision=decision,
                    reflection=reflection,
                )
            )

            if reflection.accept:
                break

            active_patches = merge_unique_patches(
                active_patches,
                reflection.prompt_patches,
            )

        final_decision = turns[-1].decision

        last_reflection = turns[-1].reflection
        if not last_reflection.accept:
            final_decision.needs_human_review = True

        if final_decision.label == "uncertain":
            final_decision.needs_human_review = True

        benchmark_similarity = None
        if pair.human_labels:
            benchmark_similarity = compare_single_prediction(
                pred_label=final_decision.label,
                human_labels=pair.human_labels,
            )

            true_label = majority_label(pair.human_labels)
            agreement = benchmark_similarity["agent_human_agreement"]

            # Learn only from clearly wrong / weak cases
            if true_label is not None:
                should_learn = (
                    final_decision.label != true_label
                    or agreement < 0.75
                    or final_decision.needs_human_review
                )

                if should_learn:
                    learning = self.policy_learner.learn_from_case(
                        pair=pair,
                        evidence=evidence,
                        decision=final_decision,
                        true_label=true_label,
                        agreement=agreement,
                    )
                    self.policy_memory.add_learning(
                        learning=learning,
                        pair_id=pair.pair_id,
                        true_label=true_label,
                    )

        return PipelineOutput(
            pair_id=pair.pair_id,
            final_decision=final_decision,
            turns=turns,
            benchmark_similarity=benchmark_similarity,
        )