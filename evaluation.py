from typing import Any, Dict, List

from schemas import PipelineOutput, ProductPair, consensus_strength, majority_label


def compare_single_prediction(pred_label: str, human_labels: List[str]) -> Dict[str, Any]:
    if not human_labels:
        return {}

    majority = majority_label(human_labels)
    strength = consensus_strength(human_labels)

    matches = 0
    for label in human_labels:
        if label == pred_label:
            matches += 1

    agent_human_agreement = matches / len(human_labels)
    majority_match = pred_label == majority

    return {
        "pred_label": pred_label,
        "majority_label": majority,
        "majority_match": majority_match,
        "agent_human_agreement": agent_human_agreement,
        "human_consensus_strength": strength,
        "human_labels": human_labels,
    }


def evaluate_outputs(pairs: List[ProductPair], outputs: List[PipelineOutput]) -> Dict[str, Any]:
    pair_map = {pair.pair_id: pair for pair in pairs}
    scored_rows: List[Dict[str, Any]] = []

    for output in outputs:
        pair = pair_map.get(output.pair_id)
        if pair is None:
            continue

        if not pair.human_labels:
            continue

        row = compare_single_prediction(
            pred_label=output.final_decision.label,
            human_labels=pair.human_labels,
        )
        row["pair_id"] = pair.pair_id
        row["needs_human_review"] = output.final_decision.needs_human_review
        scored_rows.append(row)

    n = len(scored_rows)
    if n == 0:
        return {
            "n_scored_pairs": 0,
            "majority_vote_accuracy": None,
            "avg_agent_human_agreement": None,
            "avg_human_consensus_strength": None,
            "uncertain_rate": None,
            "uncertain_when_humans_disagree_rate": None,
            "rows": [],
        }

    majority_hits = 0
    agreement_sum = 0.0
    consensus_sum = 0.0
    uncertain_count = 0
    disagree_bucket = 0
    uncertain_and_disagree = 0

    for row in scored_rows:
        if row["majority_match"]:
            majority_hits += 1

        agreement_sum += row["agent_human_agreement"]
        consensus_sum += row["human_consensus_strength"]

        pred_label = row["pred_label"]
        if pred_label == "uncertain":
            uncertain_count += 1

        if row["human_consensus_strength"] < 0.67:
            disagree_bucket += 1
            if pred_label == "uncertain":
                uncertain_and_disagree += 1

    uncertain_when_humans_disagree_rate = None
    if disagree_bucket > 0:
        uncertain_when_humans_disagree_rate = uncertain_and_disagree / disagree_bucket

    return {
        "n_scored_pairs": n,
        "majority_vote_accuracy": majority_hits / n,
        "avg_agent_human_agreement": agreement_sum / n,
        "avg_human_consensus_strength": consensus_sum / n,
        "uncertain_rate": uncertain_count / n,
        "uncertain_when_humans_disagree_rate": uncertain_when_humans_disagree_rate,
        "rows": scored_rows,
    }