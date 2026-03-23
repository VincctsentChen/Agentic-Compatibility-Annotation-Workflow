from pprint import pprint

from agents import MainAnnotatorAgent, PolicyLearnerAgent, ReflectionAgent
from evaluation import evaluate_outputs
from orchestrator import CompatibilityOrchestrator
from policy_memory import PolicyMemory
from retriever import ExampleMemory
from schemas import Product, ProductPair
from tools import FeatureBuilder
from llm_client import DashScopeQwenClient
import os


def make_product(
    product_id: str,
    title: str,
    description: str,
    category: str,
    image_summaries=None,
    metadata=None,
):
    if image_summaries is None:
        image_summaries = []
    if metadata is None:
        metadata = {}

    return Product(
        product_id=product_id,
        title=title,
        description=description,
        category=category,
        image_refs=[],
        image_summaries=image_summaries,
        metadata=metadata,
    )


def main():
    train_pairs = [
        ProductPair(
            pair_id="train_1",
            split="train",
            product_a=make_product(
                "a1",
                "Modern Walnut Dining Table",
                "Rectangular dining table in walnut finish.",
                "dining table",
                image_summaries=["wood dining table with modern lines"],
            ),
            product_b=make_product(
                "b1",
                "Upholstered Dining Chair Set",
                "Set of two dining chairs with beige fabric.",
                "dining chair",
                image_summaries=["neutral dining chairs for dining room"],
            ),
            human_labels=["compatible", "compatible", "compatible"],
        ),
        ProductPair(
            pair_id="train_2",
            split="train",
            product_a=make_product(
                "a2",
                "Ergonomic Mesh Office Chair",
                "Adjustable office chair for desk work.",
                "office chair",
                image_summaries=["black mesh chair for office desk"],
            ),
            product_b=make_product(
                "b2",
                "Luxury King Bed Frame",
                "Padded bedroom bed frame with tall headboard.",
                "bed frame",
                image_summaries=["upholstered bed in bedroom"],
            ),
            human_labels=["incompatible", "incompatible", "incompatible"],
        ),
        ProductPair(
            pair_id="train_3",
            split="train",
            product_a=make_product(
                "a3",
                "Modern Coffee Table",
                "Glass-top coffee table with black frame.",
                "coffee table",
                image_summaries=["modern black glass coffee table"],
            ),
            product_b=make_product(
                "b3",
                "Farmhouse Sofa",
                "Rustic sofa in distressed leather.",
                "sofa",
                image_summaries=["rustic brown leather sofa"],
            ),
            human_labels=["uncertain", "incompatible", "uncertain"],
        ),
    ]

    benchmark_pairs = [
        ProductPair(
            pair_id="bench_1",
            split="test",
            product_a=make_product(
                "c1",
                "Scandinavian Dining Chair",
                "Light wood dining chair with fabric seat.",
                "dining chair",
                image_summaries=["light wood dining chair"],
            ),
            product_b=make_product(
                "d1",
                "Extendable Dining Table",
                "Modern dining table for six people.",
                "dining table",
                image_summaries=["modern dining table with wood top"],
            ),
            human_labels=["compatible", "compatible", "compatible"],
        ),
        ProductPair(
            pair_id="bench_2",
            split="test",
            product_a=make_product(
                "c2",
                "Vintage Coffee Table",
                "Old-world carved wood coffee table.",
                "coffee table",
                image_summaries=["ornate carved coffee table"],
            ),
            product_b=make_product(
                "d2",
                "Freestanding Bathtub",
                "White soaking tub for bathroom.",
                "bathtub",
                image_summaries=["modern freestanding bathtub"],
            ),
            human_labels=["incompatible", "incompatible", "uncertain"],
        ),
    ]

    memory = ExampleMemory()
    memory.fit(train_pairs)

    policy_memory = PolicyMemory()

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key or not api_key.strip():
        raise SystemExit(
            "DASHSCOPE_API_KEY is not set. Export your DashScope API key before running:\n"
            "  export DASHSCOPE_API_KEY='your-key-here'\n"
            "(In zsh/bash you can add that line to ~/.zshrc or ~/.bashrc.)"
        )

    llm = DashScopeQwenClient(
        api_key=api_key.strip(),
        model="qwen3.5-flash",
        base_url="https://dashscope-us.aliyuncs.com/compatible-mode/v1",
    )
    feature_builder = FeatureBuilder()

    annotator = MainAnnotatorAgent(llm=llm, debug_prompts=True)
    reflector = ReflectionAgent(llm=llm, debug_prompts=True)
    policy_learner = PolicyLearnerAgent(llm=llm)

    orchestrator = CompatibilityOrchestrator(
        feature_builder=feature_builder,
        memory=memory,
        policy_memory=policy_memory,
        annotator=annotator,
        reflector=reflector,
        policy_learner=policy_learner,
        max_turn=3,
        retrieval_k=6,
    )

    outputs = []
    for pair in benchmark_pairs:
        out = orchestrator.annotate_pair(pair)
        outputs.append(out)

        print("=" * 80)
        print("PAIR:", out.pair_id)
        print("FINAL LABEL:", out.final_decision.label)
        print("CONFIDENCE:", out.final_decision.confidence)
        print("RATIONALE:", out.final_decision.rationale)
        print("NEEDS HUMAN REVIEW:", out.final_decision.needs_human_review)
        print("BENCHMARK SIMILARITY:")
        pprint(out.benchmark_similarity)

        print("\nTURN TRACE")
        for turn in out.turns:
            print(f"Turn {turn.turn_id}")
            print("  Patches:", turn.active_prompt_patches)
            print("  Decision:", turn.decision.label, turn.decision.confidence)
            print("  Reflection accept:", turn.reflection.accept)
            print("  Reflection issues:", turn.reflection.issues_found)

        print("\nCURRENT POLICY MEMORY")
        policy_memory.debug_dump()

    print("\n" + "=" * 80)
    print("DATASET-LEVEL EVALUATION")
    metrics = evaluate_outputs(benchmark_pairs, outputs)
    pprint(metrics)


if __name__ == "__main__":
    main()